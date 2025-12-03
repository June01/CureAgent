import sys
import json
import hashlib
import torch
from typing import List


def get_md5(input_str):
    # Create an MD5 hash object
    md5_hash = hashlib.md5()

    # Encode the string and update the hash object
    md5_hash.update(input_str.encode("utf-8"))

    # Return the hexadecimal MD5 digest
    return md5_hash.hexdigest()


def tool_result_format(function_call_messages):
    current_output = "\n\n<details>\n<summary> <strong>Verfied Feedback from Tools</strong>, click to see details:</summary>\n\n"
    for each_message in function_call_messages:
        if each_message["role"] == "tool":
            current_output += f"{each_message['content']}\n\n"
    current_output += "</details>\n\n\n"
    return current_output


class NoRepeatSentenceProcessor:
    def __init__(
        self, forbidden_sequences: List[List[int]], allowed_prefix_length: int
    ):
        """
        Args:
            forbidden_sequences (List[List[int]]): A list of token ID sequences corresponding to forbidden sentences.
            allowed_prefix_length (int): The number k such that if the generated tokens match the first k tokens
                                         of a forbidden sequence, then the candidate token that would extend the match is blocked.
        """
        self.allowed_prefix_length = allowed_prefix_length
        # Build a lookup dictionary: key is a tuple of the first k tokens, value is a set of tokens to block.
        self.forbidden_prefix_dict = {}
        for seq in forbidden_sequences:
            if len(seq) > allowed_prefix_length:
                prefix = tuple(seq[:allowed_prefix_length])
                next_token = seq[allowed_prefix_length]
                self.forbidden_prefix_dict.setdefault(prefix, set()).add(next_token)

    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        """
        Modifies the logits to block tokens that would extend a forbidden sentence.

        Args:
            token_ids (List[int]): List of token IDs generated so far.
            logits (torch.Tensor): Logits tensor for the next token (shape: [vocab_size]).

        Returns:
            torch.Tensor: Modified logits.
        """
        if len(token_ids) >= self.allowed_prefix_length:
            prefix = tuple(token_ids[: self.allowed_prefix_length])
            if prefix in self.forbidden_prefix_dict:
                for token_id in self.forbidden_prefix_dict[prefix]:
                    logits[token_id] = -float("inf")
        return logits


class ReasoningTraceChecker:
    def __init__(self, question, conversation, init_index=None):
        self.question = question
        self.conversation = conversation
        self.existing_thoughts = []
        self.existing_actions = []
        if init_index is not None:
            self.index = init_index
        else:
            self.index = 1
        self.question = self.question.lower()
        self.new_thoughts = []
        self.new_actions = []

    def check_conversation(self):
        info = ""
        current_index = self.index
        for i in range(current_index, len(self.conversation)):
            each = self.conversation[i]
            self.index = i
            if each["role"] == "assistant":
                print(each)
                thought = each["content"]
                actions = each["tool_calls"]

                good_status, current_info = self.check_repeat_thought(thought)
                info += current_info
                if not good_status:
                    return False, info

                good_status, current_info = self.check_repeat_action(actions)
                info += current_info
                if not good_status:
                    return False, info
        return True, info

    def check_repeat_thought(self, thought):
        if thought in self.existing_thoughts:
            return False, "repeat_thought"
        self.existing_thoughts.append(thought)
        return True, ""

    def check_repeat_action(self, actions):
        if type(actions) != list:
            actions = json.loads(actions)
        for each_action in actions:
            if "call_id" in each_action:
                del each_action["call_id"]
            each_action = json.dumps(each_action)
            if each_action in self.existing_actions:
                return False, "repeat_action"
            self.existing_actions.append(each_action)
        return True, ""
