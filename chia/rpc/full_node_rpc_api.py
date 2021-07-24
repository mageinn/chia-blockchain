from typing import Any, Callable, Dict, List, Optional, Tuple
from blspy import AugSchemeMPL, G1Element, G2Element
from chiapos import Verifier

from distutils.util import strtobool
from chia.consensus.block_record import BlockRecord
from chia.consensus.pos_quality import UI_ACTUAL_SPACE_CONSTANT_FACTOR
from chia.full_node.full_node import FullNode
from chia.full_node.mempool_check_conditions import get_puzzle_and_solution_for_coin
from chia.consensus.pot_iterations import calculate_ip_iters, calculate_sp_iters
from chia.types.blockchain_format.program import Program, SerializedProgram
from chia.types.blockchain_format.sized_bytes import bytes32
from chia.types.coin_record import CoinRecord
from chia.types.coin_solution import CoinSolution
from chia.types.full_block import FullBlock
from chia.types.generator_types import BlockGenerator
from chia.types.mempool_inclusion_status import MempoolInclusionStatus
from chia.types.spend_bundle import SpendBundle
from chia.types.unfinished_header_block import UnfinishedHeaderBlock
from chia.util.byte_types import hexstr_to_bytes
from chia.util.ints import uint32, uint64, uint128
from chia.util.ws_message import WsRpcMessage, create_payload_dict
from chia.pools.pool_puzzles import (
    launcher_id_to_p2_puzzle_hash, 
    solution_to_extra_data, 
    get_delayed_puz_info_from_launcher_spend,
    pool_state_to_inner_puzzle,
    create_full_puzzle,
    get_most_recent_singleton_coin_from_coin_solution
)
from chia.pools.pool_wallet_info import PoolState

class FullNodeRpcApi:
    def __init__(self, service: FullNode):
        self.service = service
        self.service_name = "chia_full_node"
        self.cached_blockchain_state: Optional[Dict] = None

    def get_routes(self) -> Dict[str, Callable]:
        return {
            # Blockchain
            "/get_blockchain_state": self.get_blockchain_state,
            "/get_block": self.get_block,
            "/get_blocks": self.get_blocks,
            "/get_block_record_by_height": self.get_block_record_by_height,
            "/get_block_record": self.get_block_record,
            "/get_block_records": self.get_block_records,
            "/get_unfinished_block_headers": self.get_unfinished_block_headers,
            "/get_network_space": self.get_network_space,
            "/get_additions_and_removals": self.get_additions_and_removals,
            "/get_initial_freeze_period": self.get_initial_freeze_period,
            "/get_network_info": self.get_network_info,
            "/get_recent_signage_point_or_eos": self.get_recent_signage_point_or_eos,
            # Singletons
            "/get_singleton_state": self.get_singleton_state,
            "/get_p2_puzzle_hash_from_launcher_id": self.get_p2_puzzle_hash_from_launcher_id,
            "/check_relative_lock_height": self.check_relative_lock_height,
            # Blspy Operations
            "/aggregate_verify_signature": self.aggregate_verify_signature,
            "/verify_signature": self.verify_signature,
            # ChiaPos Operations
            "/get_proof_quality_string": self.get_proof_quality_string,
            # Pool Stuff
            "/get_delayed_puzzle_info_from_launcher_id": self.get_delayed_puzzle_info_from_launcher_id,
            "/get_pool_state_from_coin_spend": self.get_pool_state_from_coin_spend,
            "/confirm_signage_point_or_eos": self.confirm_signage_point_or_eos,
            # Coins
            "/get_coin_records_by_puzzle_hash": self.get_coin_records_by_puzzle_hash,
            "/get_coin_records_by_puzzle_hashes": self.get_coin_records_by_puzzle_hashes,
            "/get_coin_record_by_name": self.get_coin_record_by_name,
            "/push_tx": self.push_tx,
            "/get_puzzle_and_solution": self.get_puzzle_and_solution,
            # Mempool
            "/get_all_mempool_tx_ids": self.get_all_mempool_tx_ids,
            "/get_all_mempool_items": self.get_all_mempool_items,
            "/get_mempool_item_by_tx_id": self.get_mempool_item_by_tx_id,
        }

    async def _state_changed(self, change: str) -> List[WsRpcMessage]:
        payloads = []
        if change == "new_peak" or change == "sync_mode":
            data = await self.get_blockchain_state({})
            assert data is not None
            payloads.append(
                create_payload_dict(
                    "get_blockchain_state",
                    data,
                    self.service_name,
                    "wallet_ui",
                )
            )
            return payloads
        return []
    
    #Helper Method Area
    async def get_coin_spend(self, coin_record: CoinRecord) -> Optional[CoinSolution]:
        if coin_record is None or not coin_record.spent:
            return None

        header_hash = self.service.blockchain.height_to_hash(coin_record.spent_block_index)
        block: Optional[FullBlock] = await self.service.block_store.get_full_block(header_hash)

        if block is None or block.transactions_generator is None:
            return None

        block_generator: Optional[BlockGenerator] = await self.service.blockchain.get_block_generator(block)
        assert block_generator is not None
        error, puzzle, solution = get_puzzle_and_solution_for_coin(
            block_generator, coin_record.coin.name(), self.service.constants.MAX_BLOCK_COST_CLVM
        )
        if error is not None:
            return None

        puzzle_ser: SerializedProgram = SerializedProgram.from_program(Program.to(puzzle))
        solution_ser: SerializedProgram = SerializedProgram.from_program(Program.to(solution))
        return CoinSolution(coin_record.coin, puzzle_ser, solution_ser)

    async def validate_puzzle_hash(
        self,
        launcher_id: bytes32,
        delay_ph: bytes32,
        delay_time: uint64,
        pool_state: PoolState,
        outer_puzzle_hash: bytes32,
        genesis_challenge: bytes32,
    ) -> bool:
        inner_puzzle: Program = pool_state_to_inner_puzzle(pool_state, launcher_id, genesis_challenge, delay_time, delay_ph)
        new_full_puzzle: Program = create_full_puzzle(inner_puzzle, launcher_id)
        return new_full_puzzle.get_tree_hash() == outer_puzzle_hash

    async def try_get_singleton_state(self, launcher_id, confirmation_security_threshold, has_farmer_data,   #bured solution, buried state, tip state
                                   singleton_tip, singleton_tip_state, delay_time, delay_puzzle_hash) -> Optional[Tuple[CoinSolution,PoolState,PoolState]]:
        try:
            genesis_challenge = self.service.constants.GENESIS_CHALLENGE

            peak: Optional[BlockRecord] = self.service.blockchain.get_peak()

            if peak is None:
                return None

            peak_height = peak.height;

            if peak_height == 0:
                return None

            if not has_farmer_data:
                launcher_coin: Optional[CoinRecord] = await self.service.blockchain.coin_store.get_coin_record(launcher_id)
                if launcher_coin is None:
                    return None

                last_solution = await self.get_coin_spend(launcher_coin)

                if last_solution is None:
                    return None

                delay_time, delay_puzzle_hash = get_delayed_puz_info_from_launcher_spend(last_solution)
                saved_state = solution_to_extra_data(last_solution)

                if saved_state is None:
                    return None
            else:
                last_solution = CoinSolution.from_json_dict(request["singleton_tip"])
                saved_state = PoolState.from_json_dict(request["singleton_tip_state"])
                delay_time = int(request["delay_time"])
                delay_puzzle_hash = bytes.fromhex(request["delay_puzzle_hash"])

            saved_solution = last_solution
            last_not_none_state: PoolState = saved_state
            assert last_solution is not None

            if launcher_coin is None:
                last_coin_record: Optional[CoinRecord] =  await self.service.blockchain.coin_store.get_coin_record(last_solution.coin.name())
            else:
                last_coin_record: Optional[CoinRecord] = launcher_coin

            assert last_coin_record is not None
        
            while True:
                next_coin: Optional[Coin] = get_most_recent_singleton_coin_from_coin_solution(last_solution)

                if next_coin is None:
                    return None

                next_coin_record: Optional[CoinRecord] = await self.service.blockchain.coin_store.get_coin_record(next_coin.name())
                
                if next_coin_record is None:
                    return None

                if not next_coin_record.spent:
                    if not self.validate_puzzle_hash(
                        launcher_id,
                        delay_puzzle_hash,
                        delay_time,
                        last_not_none_state,
                        next_coin_record.coin.puzzle_hash,
                        genesis_challenge,
                    ):
                        return None
                    break

                last_solution: Optional[CoinSolution] = await self.get_coin_spend(next_coin_record)
                
                if last_solution is None:
                    return None

                pool_state: Optional[PoolState] = solution_to_extra_data(last_solution)

                if pool_state is not None:
                    last_not_none_state = pool_state
                if peak_height - confirmation_security_threshold >= next_coin_record.spent_block_index:
                    # There is a state transition, and it is sufficiently buried
                    saved_solution = last_solution
                    saved_state = last_not_none_state

            return saved_solution, saved_state, last_not_none_state
        except Exception as e:
            self.service.log.exception("Exception while getting singleton state")
            return None

    #Helper Method Area End

    async def get_singleton_state(self, request: Dict):
        launcher_id = bytes.fromhex(request["launcher_id"])
        confirmation_security_threshold = int(request["confirmation_security_threshold"])
        genesis_challenge = self.service.constants.GENESIS_CHALLENGE
        has_farmer_data = strtobool(request["has_farmer_data"])

        if has_farmer_data:
            last_solution = CoinSolution.from_json_dict(request["singleton_tip"])
            saved_state = PoolState.from_json_dict(request["singleton_tip_state"])
            delay_time = int(request["delay_time"])
            delay_puzzle_hash = bytes.fromhex(request["delay_puzzle_hash"])
        else:
            last_solution = None
            saved_state = None
            delay_time = None
            delay_puzzle_hash = None


        singleton_state_tuple = await self.try_get_singleton_state(launcher_id, confirmation_security_threshold, has_farmer_data, 
                                                                   last_solution, saved_state, delay_time, delay_puzzle_hash)
        
        if singleton_state_tuple is None:
            return {"has_value":False}

        saved_solution, saved_state, tip_state = singleton_state_tuple

        return {"has_value":True, "singleton_state":{"buried_singleton_tip":saved_solution, "buried_singleton_tip_state":saved_state,"singleton_tip_state":tip_state}}

    async def aggregate_verify_signature(self, request: Dict):
        pk1: G1Element = G1Element.from_bytes(bytes.fromhex(request["plot_public_key"]))
        pk2: G1Element = G1Element.from_bytes(bytes.fromhex(request["owner_pk"]))
        m1: bytes = bytes.fromhex(request["payload_hash"])
        sig: G2Element = G2Element.from_bytes(bytes.fromhex(request["signature"]))

        valid_sig = AugSchemeMPL.aggregate_verify(
            [pk1, pk2], [m1, m1], sig
        )

        return {"valid": valid_sig}

    async def verify_signature(self, request: Dict):
        ownerPk = G1Element.from_bytes(bytes.fromhex(request["owner_pk"]))
        payloadHash = bytes.fromhex(request["payload_hash"])
        signature = G2Element.from_bytes(bytes.fromhex(request["signature"]))

        valid_sig = AugSchemeMPL.verify(ownerPk, payloadHash, signature)

        return {"valid": valid_sig}

    async def get_proof_quality_string(self, request: Dict):
        plot_id = bytes.fromhex(request["plot_id"])
        size = int(request["size"])
        challenge = bytes.fromhex(request["challenge"])
        proof = bytes.fromhex(request["proof"])
        quality_str = Verifier().validate_proof(plot_id, size, challenge, proof)
        if (quality_str is None):
            return {"valid": False }
        else:
            return {"valid": True, "quality_str": quality_str}

    async def get_p2_puzzle_hash_from_launcher_id(self, request: Dict):
        launcher_Id = bytes.fromhex(request["launcher_id"])
        delay_time = uint64(request["delay_time"])
        delay_puzzle_hash = bytes.fromhex(request["delay_puzzle_hash"])

        p2_puzzle_hash = launcher_id_to_p2_puzzle_hash(launcher_Id, delay_time, delay_puzzle_hash)
        return {"p2_puzzle_hash": p2_puzzle_hash.hex()}

    async def check_relative_lock_height(self, request: Dict):
        coin_id = bytes.fromhex(request["coin_id"])      
        relative_lock_height = int(request["relative_lock_height"])
        coin_record = await self.service.blockchain.coin_store.get_coin_record(coin_id)

        if (coin_record is None):
            return {"valid": False}

        peak: Optional[BlockRecord] = self.service.blockchain.get_peak()
        peak_height = peak.height
        valid = peak_height - coin_record.confirmed_block_index > relative_lock_height

        return {"valid": valid}

    async def get_delayed_puzzle_info_from_launcher_id(self, request: Dict):
        launcher_id = bytes.fromhex(request["launcher_id"])
        coin_record = await self.service.blockchain.coin_store.get_coin_record(launcher_id)

        if (coin_record is None):
            return {"has_value": False}

        coin_sol = await self.get_coin_spend(coin_record);

        if (coin_sol is None):
            return {"has_value": False}

        seconds, delayed_puzzle_hash = get_delayed_puz_info_from_launcher_spend(coin_sol)

        return {"has_value":True, "seconds": seconds, "delayed_puzzle_hash": delayed_puzzle_hash}
            
    async def get_pool_state_from_coin_spend(self, request: Dict):
        coin_sol = CoinSolution.from_json_dict(request)
        pool_state = solution_to_extra_data(coin_sol)
        has_value = pool_state != None
        
        return {"has_value":has_value, "pool_state":pool_state}

    async def confirm_signage_point_or_eos(self, request: Dict):
        search_range = 100

        sp_hash: bytes32 = hexstr_to_bytes(request["sp_hash"])
        hint_height: int = int(request["hint_height"])
        rc_challenge: bytes32 = hexstr_to_bytes(request["rc_challenge"])
        cc_iters: int = int(request["cc_iters"])

        # If it's still in the full node store, it's not reverted
        if self.service.full_node_store.get_signage_point(sp_hash):
            return {"valid": True}

        # Otherwise we can backtrack from peak to find it in the blockchain
        hint_b_hash: Optional[bytes32] = self.service.blockchain.height_to_hash(hint_height)

        if (hint_b_hash is None):
            self.service.log.warning("hint_b_hash is None")
            return {"valid": False} #This should never happen because of the processing delay but you never know

        next_b: Optional[BlockRecord] = self.service.blockchain.try_block_record(hint_b_hash)

        if (next_b is None):
            self.service.log.warning("next_b is None")
            return {"valid": False} #This should never happen because of the processing delay but you never know

        curr_b: Optional[BlockRecord] = self.service.blockchain.try_block_record(next_b.prev_hash)
        if (curr_b is None):
            self.service.log.warning("curr_b is None")
            return {"valid": False} #This should never happen because of the processing delay but you never know

        for _ in range(search_range):
            sp_total_iters = cc_iters + curr_b.ip_sub_slot_total_iters(self.service.constants) #Total iters at start of SP
            if curr_b.reward_infusion_new_challenge == rc_challenge:

                next_b_total_iters = next_b.total_iters #Total Iters when fully infused

                self.service.log.warning("cc_iters " + str(cc_iters) +  " next_b_total_iters " + str(next_b_total_iters) + " sp_total_iters " + str(sp_total_iters))

                return {"valid": next_b_total_iters >= sp_total_iters}
                
            if curr_b.finished_reward_slot_hashes is not None:
                assert curr_b.finished_challenge_slot_hashes is not None
                for eos_rc in curr_b.finished_reward_slot_hashes:
                    if eos_rc == rc_challenge:
                        sp_total_iters = calculate_sp_iters(self.service.constants, next_b.sub_slot_iters, next_b.signage_point_index + 1)
                        next_b_total_iters = next_b.total_iters

                        self.service.log.warning("cc_iters " + str(cc_iters) +  " next_b_total_iters " + str(next_b_total_iters) + " sp_total_iters " + str(sp_total_iters))

                        return {"valid": next_b_total_iters >= sp_total_iters}

            next_b = curr_b
            curr_b_optional = self.service.blockchain.try_block_record(curr_b.prev_hash)
            if curr_b_optional is None:
                break
            curr_b = curr_b_optional

        self.service.log.warning("end of range or start of chain")
        return {"valid": False}

    async def get_initial_freeze_period(self, _: Dict):
        freeze_period = self.service.constants.INITIAL_FREEZE_END_TIMESTAMP
        return {"INITIAL_FREEZE_END_TIMESTAMP": freeze_period}

    async def get_blockchain_state(self, _request: Dict):
        """
        Returns a summary of the node's view of the blockchain.
        """
        if self.service.initialized is False:
            res: Dict = {
                "blockchain_state": {
                    "peak": None,
                    "genesis_challenge_initialized": self.service.initialized,
                    "sync": {
                        "sync_mode": False,
                        "synced": False,
                        "sync_tip_height": 0,
                        "sync_progress_height": 0,
                    },
                    "difficulty": 0,
                    "sub_slot_iters": 0,
                    "space": 0,
                    "mempool_size": 0,
                },
            }
            return res
        peak: Optional[BlockRecord] = self.service.blockchain.get_peak()

        if peak is not None and peak.height > 0:
            difficulty = uint64(peak.weight - self.service.blockchain.block_record(peak.prev_hash).weight)
            sub_slot_iters = peak.sub_slot_iters
        else:
            difficulty = self.service.constants.DIFFICULTY_STARTING
            sub_slot_iters = self.service.constants.SUB_SLOT_ITERS_STARTING

        sync_mode: bool = self.service.sync_store.get_sync_mode() or self.service.sync_store.get_long_sync()

        sync_tip_height: Optional[uint32] = uint32(0)
        if sync_mode:
            if self.service.sync_store.get_sync_target_height() is not None:
                sync_tip_height = self.service.sync_store.get_sync_target_height()
                assert sync_tip_height is not None
            if peak is not None:
                sync_progress_height: uint32 = peak.height
                # Don't display we're syncing towards 0, instead show 'Syncing height/height'
                # until sync_store retrieves the correct number.
                if sync_tip_height == uint32(0):
                    sync_tip_height = peak.height
            else:
                sync_progress_height = uint32(0)
        else:
            sync_progress_height = uint32(0)

        if peak is not None and peak.height > 1:
            newer_block_hex = peak.header_hash.hex()
            # Average over the last day
            header_hash = self.service.blockchain.height_to_hash(uint32(max(1, peak.height - 4608)))
            assert header_hash is not None
            older_block_hex = header_hash.hex()
            space = await self.get_network_space(
                {"newer_block_header_hash": newer_block_hex, "older_block_header_hash": older_block_hex}
            )
        else:
            space = {"space": uint128(0)}

        if self.service.mempool_manager is not None:
            mempool_size = len(self.service.mempool_manager.mempool.spends)
        else:
            mempool_size = 0
        if self.service.server is not None:
            is_connected = len(self.service.server.get_full_node_connections()) > 0
        else:
            is_connected = False
        synced = await self.service.synced() and is_connected

        assert space is not None
        response: Dict = {
            "blockchain_state": {
                "peak": peak,
                "genesis_challenge_initialized": self.service.initialized,
                "sync": {
                    "sync_mode": sync_mode,
                    "synced": synced,
                    "sync_tip_height": sync_tip_height,
                    "sync_progress_height": sync_progress_height,
                },
                "difficulty": difficulty,
                "sub_slot_iters": sub_slot_iters,
                "space": space["space"],
                "mempool_size": mempool_size,
            },
        }
        self.cached_blockchain_state = dict(response["blockchain_state"])
        return response

    async def get_network_info(self, request: Dict):
        network_name = self.service.config["selected_network"]
        address_prefix = self.service.config["network_overrides"]["config"][network_name]["address_prefix"]
        return {"network_name": network_name, "network_prefix": address_prefix}

    async def get_recent_signage_point_or_eos(self, request: Dict):
        peak = self.service.blockchain.get_peak();
        if "sp_hash" not in request:
            challenge_hash: bytes32 = hexstr_to_bytes(request["challenge_hash"])
            # This is the case of getting an end of slot
            eos_tuple = self.service.full_node_store.recent_eos.get(challenge_hash)
            if not eos_tuple:
                raise ValueError(f"Did not find eos {challenge_hash.hex()} in cache")
            eos, time_received = eos_tuple

            # If it's still in the full node store, it's not reverted
            if self.service.full_node_store.get_sub_slot(eos.challenge_chain.get_hash()):
                return {"eos": eos, "time_received": time_received, "reverted": False, "peak_height": peak.height}

            # Otherwise we can backtrack from peak to find it in the blockchain
            curr: Optional[BlockRecord] = peak
            if curr is None:
                raise ValueError("No blocks in the chain")
            
            number_of_slots_searched = 0
            while number_of_slots_searched < 10:
                if curr.first_in_sub_slot:
                    assert curr.finished_challenge_slot_hashes is not None
                    if curr.finished_challenge_slot_hashes[-1] == eos.challenge_chain.get_hash():
                        # Found this slot in the blockchain
                        return {"eos": eos, "time_received": time_received, "reverted": False, "peak_height": peak.height}
                    number_of_slots_searched += len(curr.finished_challenge_slot_hashes)
                curr = self.service.blockchain.try_block_record(curr.prev_hash)
                if curr is None:
                    # Got to the beginning of the blockchain without finding the slot
                    return {"eos": eos, "time_received": time_received, "reverted": True, "peak_height": peak.height}

            # Backtracked through 10 slots but still did not find it
            return {"eos": eos, "time_received": time_received, "reverted": True, "peak_height": peak.height}

        # Now we handle the case of getting a signage point
        sp_hash: bytes32 = hexstr_to_bytes(request["sp_hash"])
        sp_tuple = self.service.full_node_store.recent_signage_points.get(sp_hash)
        if sp_tuple is None:
            raise ValueError(f"Did not find sp {sp_hash.hex()} in cache")

        sp, time_received = sp_tuple

        # If it's still in the full node store, it's not reverted
        if self.service.full_node_store.get_signage_point(sp_hash):
            return {"signage_point": sp, "time_received": time_received, "reverted": False, "peak_height": peak.height}

        # Otherwise we can backtrack from peak to find it in the blockchain
        rc_challenge: bytes32 = sp.rc_vdf.challenge
        next_b: Optional[BlockRecord] = None
        curr_b_optional: Optional[BlockRecord] = peak
        assert curr_b_optional is not None
        curr_b: BlockRecord = curr_b_optional

        for _ in range(200):
            sp_total_iters = sp.cc_vdf.number_of_iterations + curr_b.ip_sub_slot_total_iters(self.service.constants)
            if curr_b.reward_infusion_new_challenge == rc_challenge:
                if next_b is None:
                    return {"signage_point": sp, "time_received": time_received, "reverted": False, "peak_height": peak.height}
                next_b_total_iters = next_b.ip_sub_slot_total_iters(self.service.constants) + next_b.ip_iters( #Iters when fully infused
                    self.service.constants
                )

                return {
                    "signage_point": sp,
                    "time_received": time_received,
                    "reverted": sp_total_iters > next_b_total_iters,
                    "peak_height": peak.height
                }
            if curr_b.finished_reward_slot_hashes is not None:
                assert curr_b.finished_challenge_slot_hashes is not None
                for eos_rc in curr_b.finished_challenge_slot_hashes:
                    if eos_rc == rc_challenge:
                        if next_b is None:
                            return {"signage_point": sp, "time_received": time_received, "reverted": False, "peak_height": peak.height}
                        next_b_total_iters = next_b.ip_sub_slot_total_iters(self.service.constants) + next_b.ip_iters(
                            self.service.constants
                        )
                        return {
                            "signage_point": sp,
                            "time_received": time_received,
                            "reverted": sp_total_iters > next_b_total_iters,
                            "peak_height": peak.height
                        }
            next_b = curr_b
            curr_b_optional = self.service.blockchain.try_block_record(curr_b.prev_hash)
            if curr_b_optional is None:
                break
            curr_b = curr_b_optional

        return {"signage_point": sp, "time_received": time_received, "reverted": True, "peak_height": peak.height}

    async def get_block(self, request: Dict) -> Optional[Dict]:
        if "header_hash" not in request:
            raise ValueError("No header_hash in request")
        header_hash = hexstr_to_bytes(request["header_hash"])

        block: Optional[FullBlock] = await self.service.block_store.get_full_block(header_hash)
        if block is None:
            raise ValueError(f"Block {header_hash.hex()} not found")

        return {"block": block}

    async def get_blocks(self, request: Dict) -> Optional[Dict]:
        if "start" not in request:
            raise ValueError("No start in request")
        if "end" not in request:
            raise ValueError("No end in request")
        exclude_hh = False
        if "exclude_header_hash" in request:
            exclude_hh = request["exclude_header_hash"]

        start = int(request["start"])
        end = int(request["end"])
        block_range = []
        for a in range(start, end):
            block_range.append(uint32(a))
        blocks: List[FullBlock] = await self.service.block_store.get_full_blocks_at(block_range)
        json_blocks = []
        for block in blocks:
            json = block.to_json_dict()
            if not exclude_hh:
                json["header_hash"] = block.header_hash.hex()
            json_blocks.append(json)
        return {"blocks": json_blocks}

    async def get_block_records(self, request: Dict) -> Optional[Dict]:
        if "start" not in request:
            raise ValueError("No start in request")
        if "end" not in request:
            raise ValueError("No end in request")

        start = int(request["start"])
        end = int(request["end"])
        records = []
        peak_height = self.service.blockchain.get_peak_height()
        if peak_height is None:
            raise ValueError("Peak is None")

        for a in range(start, end):
            if peak_height < uint32(a):
                self.service.log.warning("requested block is higher than known peak ")
                break
            header_hash: bytes32 = self.service.blockchain.height_to_hash(uint32(a))
            record: Optional[BlockRecord] = self.service.blockchain.try_block_record(header_hash)
            if record is None:
                # Fetch from DB
                record = await self.service.blockchain.block_store.get_block_record(header_hash)
            if record is None:
                raise ValueError(f"Block {header_hash.hex()} does not exist")

            records.append(record)
        return {"block_records": records}

    async def get_block_record_by_height(self, request: Dict) -> Optional[Dict]:
        if "height" not in request:
            raise ValueError("No height in request")
        height = request["height"]
        header_height = uint32(int(height))
        peak_height = self.service.blockchain.get_peak_height()
        if peak_height is None or header_height > peak_height:
            raise ValueError(f"Block height {height} not found in chain")
        header_hash: Optional[bytes32] = self.service.blockchain.height_to_hash(header_height)
        if header_hash is None:
            raise ValueError(f"Block hash {height} not found in chain")
        record: Optional[BlockRecord] = self.service.blockchain.try_block_record(header_hash)
        if record is None:
            # Fetch from DB
            record = await self.service.blockchain.block_store.get_block_record(header_hash)
        if record is None:
            raise ValueError(f"Block {header_hash} does not exist")
        return {"block_record": record}

    async def get_block_record(self, request: Dict):
        if "header_hash" not in request:
            raise ValueError("header_hash not in request")
        header_hash_str = request["header_hash"]
        header_hash = hexstr_to_bytes(header_hash_str)
        record: Optional[BlockRecord] = self.service.blockchain.try_block_record(header_hash)
        if record is None:
            # Fetch from DB
            record = await self.service.blockchain.block_store.get_block_record(header_hash)
        if record is None:
            raise ValueError(f"Block {header_hash.hex()} does not exist")

        return {"block_record": record}

    async def get_unfinished_block_headers(self, request: Dict) -> Optional[Dict]:

        peak: Optional[BlockRecord] = self.service.blockchain.get_peak()
        if peak is None:
            return {"headers": []}

        response_headers: List[UnfinishedHeaderBlock] = []
        for ub_height, block, _ in (self.service.full_node_store.get_unfinished_blocks()).values():
            if ub_height == peak.height:
                unfinished_header_block = UnfinishedHeaderBlock(
                    block.finished_sub_slots,
                    block.reward_chain_block,
                    block.challenge_chain_sp_proof,
                    block.reward_chain_sp_proof,
                    block.foliage,
                    block.foliage_transaction_block,
                    b"",
                )
                response_headers.append(unfinished_header_block)
        return {"headers": response_headers}

    async def get_network_space(self, request: Dict) -> Optional[Dict]:
        """
        Retrieves an estimate of total space validating the chain
        between two block header hashes.
        """
        if "newer_block_header_hash" not in request or "older_block_header_hash" not in request:
            raise ValueError("Invalid request. newer_block_header_hash and older_block_header_hash required")
        newer_block_hex = request["newer_block_header_hash"]
        older_block_hex = request["older_block_header_hash"]

        if newer_block_hex == older_block_hex:
            raise ValueError("New and old must not be the same")

        newer_block_bytes = hexstr_to_bytes(newer_block_hex)
        older_block_bytes = hexstr_to_bytes(older_block_hex)

        newer_block = await self.service.block_store.get_block_record(newer_block_bytes)
        if newer_block is None:
            raise ValueError("Newer block not found")
        older_block = await self.service.block_store.get_block_record(older_block_bytes)
        if older_block is None:
            raise ValueError("Newer block not found")
        delta_weight = newer_block.weight - older_block.weight

        delta_iters = newer_block.total_iters - older_block.total_iters
        weight_div_iters = delta_weight / delta_iters
        additional_difficulty_constant = self.service.constants.DIFFICULTY_CONSTANT_FACTOR
        eligible_plots_filter_multiplier = 2 ** self.service.constants.NUMBER_ZERO_BITS_PLOT_FILTER
        network_space_bytes_estimate = (
            UI_ACTUAL_SPACE_CONSTANT_FACTOR
            * weight_div_iters
            * additional_difficulty_constant
            * eligible_plots_filter_multiplier
        )
        return {"space": uint128(int(network_space_bytes_estimate))}

    async def get_coin_records_by_puzzle_hash(self, request: Dict) -> Optional[Dict]:
        """
        Retrieves the coins for a given puzzlehash, by default returns unspent coins.
        """
        if "puzzle_hash" not in request:
            raise ValueError("Puzzle hash not in request")
        kwargs: Dict[str, Any] = {"include_spent_coins": False, "puzzle_hash": hexstr_to_bytes(request["puzzle_hash"])}
        if "start_height" in request:
            kwargs["start_height"] = uint32(request["start_height"])
        if "end_height" in request:
            kwargs["end_height"] = uint32(request["end_height"])

        if "include_spent_coins" in request:
            kwargs["include_spent_coins"] = request["include_spent_coins"]

        coin_records = await self.service.blockchain.coin_store.get_coin_records_by_puzzle_hash(**kwargs)

        return {"coin_records": coin_records}

    async def get_coin_records_by_puzzle_hashes(self, request: Dict) -> Optional[Dict]:
        """
        Retrieves the coins for a given puzzlehash, by default returns unspent coins.
        """
        if "puzzle_hashes" not in request:
            raise ValueError("Puzzle hashes not in request")
        kwargs: Dict[str, Any] = {
            "include_spent_coins": False,
            "puzzle_hashes": [hexstr_to_bytes(ph) for ph in request["puzzle_hashes"]],
        }
        if "start_height" in request:
            kwargs["start_height"] = uint32(request["start_height"])
        if "end_height" in request:
            kwargs["end_height"] = uint32(request["end_height"])

        if "include_spent_coins" in request:
            kwargs["include_spent_coins"] = request["include_spent_coins"]

        coin_records = await self.service.blockchain.coin_store.get_coin_records_by_puzzle_hashes(**kwargs)

        return {"coin_records": coin_records}

    async def get_coin_record_by_name(self, request: Dict) -> Optional[Dict]:
        """
        Retrieves a coin record by it's name.
        """
        if "name" not in request:
            raise ValueError("Name not in request")
        name = hexstr_to_bytes(request["name"])

        coin_record: Optional[CoinRecord] = await self.service.blockchain.coin_store.get_coin_record(name)
        if coin_record is None:
            raise ValueError(f"Coin record 0x{name.hex()} not found")

        return {"coin_record": coin_record}

    async def push_tx(self, request: Dict) -> Optional[Dict]:
        if "spend_bundle" not in request:
            raise ValueError("Spend bundle not in request")

        spend_bundle = SpendBundle.from_json_dict(request["spend_bundle"])
        spend_name = spend_bundle.name()

        if self.service.mempool_manager.get_spendbundle(spend_name) is not None:
            status = MempoolInclusionStatus.SUCCESS
            error = None
        else:
            status, error = await self.service.respond_transaction(spend_bundle, spend_name)
            if status != MempoolInclusionStatus.SUCCESS:
                if self.service.mempool_manager.get_spendbundle(spend_name) is not None:
                    # Already in mempool
                    status = MempoolInclusionStatus.SUCCESS
                    error = None

        if status == MempoolInclusionStatus.FAILED:
            assert error is not None
            raise ValueError(f"Failed to include transaction {spend_name}, error {error.name}")
        return {
            "status": status.name,
        }

    async def get_puzzle_and_solution(self, request: Dict) -> Optional[Dict]:
        coin_name: bytes32 = hexstr_to_bytes(request["coin_id"])
        height = request["height"]
        coin_record = await self.service.coin_store.get_coin_record(coin_name)
        if coin_record is None or not coin_record.spent or coin_record.spent_block_index != height:
            raise ValueError(f"Invalid height {height}. coin record {coin_record}")

        header_hash = self.service.blockchain.height_to_hash(height)
        block: Optional[FullBlock] = await self.service.block_store.get_full_block(header_hash)

        if block is None or block.transactions_generator is None:
            raise ValueError("Invalid block or block generator")

        block_generator: Optional[BlockGenerator] = await self.service.blockchain.get_block_generator(block)
        assert block_generator is not None
        error, puzzle, solution = get_puzzle_and_solution_for_coin(
            block_generator, coin_name, self.service.constants.MAX_BLOCK_COST_CLVM
        )
        if error is not None:
            raise ValueError(f"Error: {error}")

        puzzle_ser: SerializedProgram = SerializedProgram.from_program(Program.to(puzzle))
        solution_ser: SerializedProgram = SerializedProgram.from_program(Program.to(solution))
        return {"coin_solution": CoinSolution(coin_record.coin, puzzle_ser, solution_ser)}

    async def get_additions_and_removals(self, request: Dict) -> Optional[Dict]:
        if "header_hash" not in request:
            raise ValueError("No header_hash in request")
        header_hash = hexstr_to_bytes(request["header_hash"])

        block: Optional[FullBlock] = await self.service.block_store.get_full_block(header_hash)
        if block is None:
            raise ValueError(f"Block {header_hash.hex()} not found")

        async with self.service.blockchain.lock:
            if self.service.blockchain.height_to_hash(block.height) != header_hash:
                raise ValueError(f"Block at {header_hash.hex()} is no longer in the blockchain (it's in a fork)")
            additions: List[CoinRecord] = await self.service.coin_store.get_coins_added_at_height(block.height)
            removals: List[CoinRecord] = await self.service.coin_store.get_coins_removed_at_height(block.height)

        return {"additions": additions, "removals": removals}

    async def get_all_mempool_tx_ids(self, request: Dict) -> Optional[Dict]:
        ids = list(self.service.mempool_manager.mempool.spends.keys())
        return {"tx_ids": ids}

    async def get_all_mempool_items(self, request: Dict) -> Optional[Dict]:
        spends = {}
        for tx_id, item in self.service.mempool_manager.mempool.spends.items():
            spends[tx_id.hex()] = item
        return {"mempool_items": spends}

    async def get_mempool_item_by_tx_id(self, request: Dict) -> Optional[Dict]:
        if "tx_id" not in request:
            raise ValueError("No tx_id in request")
        tx_id: bytes32 = hexstr_to_bytes(request["tx_id"])

        item = self.service.mempool_manager.get_mempool_item(tx_id)
        if item is None:
            raise ValueError(f"Tx id 0x{tx_id.hex()} not in the mempool")

        return {"mempool_item": item}
