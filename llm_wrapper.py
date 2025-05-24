import random
import time
import traceback

from llama_cpp import Llama
import torch
import torch.nn.functional as F


class BaseLLM:
    def query_tokens(self, sequence: list[int], top_k: int) -> (list[int], list[float], list[float]):
        """
        Query the model with a sequence of tokens and return the top-k tokens and their probabilities.

        :param sequence: List of token IDs.
        :param top_k: Number of top tokens to return.
        :return: Tuple of top token indices and their probabilities.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def tokens_to_str(self, token_sequence: list[int]) -> str:
        """
        Convert a sequence of tokens to a string.

        :param token_sequence: List of token IDs.
        :return: Decoded string.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def str_to_tokens(self, string: str, add_bos: bool = True) -> list[int]:
        """
        Convert a string to a sequence of tokens.

        :param string: Input string.
        :param add_bos: Whether to add the beginning-of-sequence token.
        :return: List of token IDs.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def token_to_alpha_order(self, token):
        """
        Convert a token to its alphabetical order.
        :param token:
        :return:
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class LlamaCppLLM(BaseLLM):
    """
    A wrapper for the llama_cpp library to provide a simplified interface for loading and using LLaMA models.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the Llama class.
        """
        self.llm = Llama(*args, **kwargs)

        self.token_dict = self._generate_token_dictionary()
        self.alpha_dict = self._generate_token_dict_alpha(self.token_dict)
        self._cached_tokens: list[int] = []

    # @timed("query_tokens")
    def query_tokens(self, sequence: list[int], top_k: int = -1) -> (list[int], list[float], list[float]):
        # Check how many tokens of the provided sequence match with the cached tokens in the LLM.
        # Then, where they first differ is where we need to rewind.
        # This can be done by setting self.n_tokens to the number of perfectly matching tokens start from the beginning
        # and then removing the tokens that match from the provided input sequence.
        # We have to maintain our own cached tokens, because the LLM does not provide a way to get the current token
        # sequence.

        # Since the debugger fucks up the LLM we need to use a bunch of print statements. FML.
        # print(f"Before:\nInput sequence: (len={len(sequence)}) {sequence}\nCached tokens: (len={len(self._cached_tokens)}){self._cached_tokens}")

        # Check if the sequence is empty
        if len(sequence) == 0:
            self._cached_tokens = sequence
            self.llm.n_tokens = 0

        # Otherwise, check how many tokens match
        else:
            cached_length = len(self._cached_tokens)
            sequence_length = len(sequence)

            # We add the '-1' because if all the tokens match, we don't want to give the LLM an empty sequence.
            matching_token_count = sequence_length-1
            for i in range(sequence_length-1):
                if i >= cached_length or sequence[i] != self._cached_tokens[i]:
                    matching_token_count = i
                    break

            # Since the LLM doesn't use our own version of the cache, we can simply assign it to the provided sequence,
            # since this is what it will become after the eval.
            self._cached_tokens = sequence
            self.llm.n_tokens = matching_token_count
            sequence = sequence[matching_token_count:]

        # print(f"After:\nInput sequence: (len={len(sequence)}) {sequence}\nCached tokens: (len={len(self._cached_tokens)}){self._cached_tokens}")
        # print(f"Reusing {self.llm.n_tokens} tokens from the cache.")

        self.llm.eval(sequence)
        last_logits = self.llm.scores[self.llm.n_tokens - 1]
        last_logits_tensor = torch.tensor(last_logits, dtype=torch.float32)
        probs = F.softmax(last_logits_tensor, dim=0)

        # TODO: If we are getting all of the tokens, we should not use topk, but just return the whole list.
        #       Perhaps we should create a separate function for this so that we don't have to return a
        #       redundant list of indices (since they'll just be [0, 1, 2, ... n_vocab-1])
        if top_k == -1:
            top_k = self.llm.n_vocab()

        topk = torch.topk(probs, top_k)

        # Get the top-k token indices and their probabilities
        # and convert them to regular Python lists
        top_indices = topk.indices.tolist()
        top_probs = topk.values.tolist()
        top_logits = last_logits_tensor[top_indices].tolist()

        return top_indices, top_probs, top_logits

    def tokens_to_str(self, token_sequence: list[int]) -> str:
        try:
            return (self.llm.detokenize(token_sequence, None, True)
                    .decode("utf-8", errors="ignore"))
        except UnicodeDecodeError:
            return '<err>'

    def str_to_tokens(self, string: str, add_bos: bool = True) -> list[int]:
        try:
            return (self.llm.tokenize(
                string.encode("utf-8", errors="ignore"), add_bos=add_bos, special=True))
        except UnicodeEncodeError:
            return [self.llm.token_bos()] if add_bos else [self.llm.token_nl()]

    def _generate_token_dictionary(self) -> dict[int, str]:
        """Generate a dictionary containing a mapping of token IDs to strings."""
        # Generate a list of token IDs
        _dict = {}
        token_list = [0]
        for i in range(self.llm.n_vocab()):
            token_list[0] = i
            token_str = self.tokens_to_str(token_list)
            _dict[i] = token_str

        return _dict

    # Create a dictionary that maps token IDs to indices of the alphabetical order of their string representation
    @staticmethod
    def _generate_token_dict_alpha(token_to_str_dict: dict[int, str]) -> dict[int, int]:
        """Generate a dictionary containing a mapping of token IDs to their alphabetical order."""
        # Create a list of tuples (token ID, token string)
        token_list = [(token_id, token_str) for token_id, token_str in token_to_str_dict.items()]

        # Sort the list based on the token strings
        sorted_token_list = sorted(token_list, key=lambda x: x[1])

        # Create a dictionary mapping token IDs to their alphabetical order
        return {token_id: index for index, (token_id, _) in enumerate(sorted_token_list)}

    def token_to_alpha_order(self, token):
        """
        Convert a token to its alphabetical order.
        :param token:
        :return:
        """
        # Check if the token is in the dictionary
        if token in self.alpha_dict:
            return self.alpha_dict[token]
        else:
            # If not, return -1 indicating it's not found
            return -1


class FallbackFakeLLM(BaseLLM):
    """
    A fallback class that simulates the behavior of a language model for testing purposes.
    """

    def __init__(self):
        self.token_dict = {
            0: "<unk>",
            1: "Apple ",
            2: "Banana ",
            3: "Cherry ",
            4: "Date ",
            5: "Elderberry ",
            6: "Fig "
        }
        self.str_to_token = {
            "Apple": 1,
            "Banana": 2,
            "Cherry": 3,
            "Date": 4,
            "Elderberry": 5,
            "Fig": 6
        }
        self.alpha_dict = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6
        }

    def query_tokens(self, sequence: list[int], top_k: int) -> (list[int], list[float]):
        tokens = torch.tensor([
                0,
                1,
                2,
                3,
                4,
                5,
                6])

        # generate random logits between -10 and 10 for each token
        logits = torch.tensor([random.uniform(-10, 10) for _ in range(len(tokens))])

        probs = F.softmax(logits, dim=0)

        # Simulate latency
        time.sleep(0.2)

        return tokens.tolist(), probs.tolist(), logits.tolist()

    def tokens_to_str(self, token_sequence: list[int]) -> str:
        return "".join([self.token_dict[token] for token in token_sequence])

    def str_to_tokens(self, string: str, add_bos: bool = True) -> list[int]:
        # return [self.token_dict[token] for token in string.split()]
        # We need to use a safe get, because the token might not be in the dictionary, use 0 for unknown tokens
        return [self.str_to_token.get(token, 0) for token in string.split()]

    def token_to_alpha_order(self, token):

        # Items are already in alphabetical order.
        return token


def get_llm(*args, **kwargs) -> BaseLLM:
    try:
        # Attempt to load the Llama model
        return LlamaCppLLM(*args, **kwargs)
    except Exception as e:
        print(f"Failed to load Llama model: {e}")
        traceback.print_exc()
        # Fallback to the fake LLM
        return FallbackFakeLLM()
