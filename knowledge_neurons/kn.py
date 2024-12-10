import json
import math
from functools import partial
from typing import Optional, Tuple

import einops
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from .patch import *


class KnowledgeNeurons:
    def __init__(
            self,
            model: nn.Module,
            tokenizer: PreTrainedTokenizerBase,
            model_type: str = "bert",
            device: str = None,
    ):
        self.model = model
        self.model_type = model_type
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # self.model.to(self.device)
        self.tokenizer = tokenizer

        self.baseline_activations = None
        if self.model_type == "bert":
            self.transformer_layers_attr = "bert.encoder.layer"
            self.input_ff_attr = "intermediate"
            self.output_ff_attr = "output.dense.weight"
            self.word_embeddings_attr = "bert.embeddings.word_embeddings.weight"
            self.unk_token = getattr(self.tokenizer, "unk_token_id", None)
        elif "gpt" in self.model_type:
            self.model_type = 'gpt'
            self.transformer_layers_attr = "transformer.h"
            self.input_ff_attr = "mlp.c_fc"
            self.output_ff_attr = "mlp.c_proj.weight"
            self.word_embeddings_attr = "transformer.wpe"
            # add pad token
            # new_tokens = ['<PAD>']
            # self.tokenizer.add_tokens(new_tokens, special_tokens=True)
            # self.model.resize_token_embeddings(len(tokenizer))
            # self.tokenizer.pad_token = '<PAD>'
        elif self.model_type == 'bart_encoder':
            self.transformer_layers_attr = "model.encoder.layers"
            self.input_ff_attr = "fc1"
            self.output_ff_attr = "fc2.weight"
            self.word_embeddings_attr = "model.encoder.embed_tokens.weight"

        elif self.model_type == 'bart_decoder':
            self.transformer_layers_attr = "model.decoder.layers"
            self.input_ff_attr = "fc1"
            self.output_ff_attr = "fc2.weight"
            self.word_embeddings_attr = "model.decoder.embed_tokens.weight"
        elif self.model_type == 'llama':
            self.model_type = 'llama'
            self.transformer_layers_attr = "model.layers"
            self.input_ff_attr = "mlp.gate_proj"  # or "mlp.up_proj", depending on the specific component you need
            self.output_ff_attr = "mlp.down_proj.weight"
            self.word_embeddings_attr = "model.embed_tokens"

        else:
            raise NotImplementedError

    def _get_output_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.output_ff_attr,
        )

    def _get_input_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

    def _get_word_embeddings(self):
        return get_attributes(self.model, self.word_embeddings_attr)

    def _get_transformer_layers(self):
        return get_attributes(self.model, self.transformer_layers_attr)

    def _prepare_inputs(self, prompt, target=None):
        # Remove the placeholder _X_ from the prompt for GPT models
        if 'MASK' in prompt:
            prompt = prompt.replace("[MASK] .", "").strip()
        elif '_X_' in prompt:
            prompt = prompt.replace("_X_.", "").strip()
        # Encode the modified prompt
        encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # For GPT models, the model is expected to predict the next word(s) after the prompt
        mask_idx = -1

        # Encode the target (ground_truth) if provided
        if target is not None:
            target_label = self.tokenizer.encode(target)
        else:
            target_label = None

        return encoded_input, mask_idx, target_label, prompt

    def _generate(self, prompt, ground_truth):
        encoded_input, mask_idx, target_label, prompt = self._prepare_inputs(
            prompt, ground_truth
        )
        n_sampling_steps = len(target_label)
        all_gt_probs = []
        all_argmax_probs = []
        argmax_tokens = []
        argmax_completion_str = ""

        for i in range(n_sampling_steps):
            if i > 0:
                # retokenize new inputs
                encoded_input, mask_idx, target_label, prompt = self._prepare_inputs(
                    prompt, ground_truth
                )
            outputs = self.model(**encoded_input)
            probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
            target_idx = target_label[i]
            gt_prob = probs[:, target_idx].item()
            all_gt_probs.append(gt_prob)

            # get info about argmax completion
            argmax_prob, argmax_id = [i.item() for i in probs.max(dim=-1)]
            argmax_tokens.append(argmax_id)
            argmax_str = self.tokenizer.decode([argmax_id])
            all_argmax_probs.append(argmax_prob)

            prompt += argmax_str
            argmax_completion_str += argmax_str

        gt_prob = math.prod(all_gt_probs) if len(all_gt_probs) > 1 else all_gt_probs[0]
        argmax_prob = (
            math.prod(all_argmax_probs)
            if len(all_argmax_probs) > 1
            else all_argmax_probs[0]
        )
        return gt_prob, argmax_prob, argmax_completion_str, argmax_tokens


    def _get_answer_str(self, prompt, ground_truth):
        encoded_input, mask_idx, target_label, prompt = self._prepare_inputs(
            prompt, ground_truth
        )
        n_sampling_steps = len(target_label)
        argmax_completion_str = ""

        for i in range(n_sampling_steps):
            if i > 0:
                encoded_input, mask_idx, target_label, prompt = self._prepare_inputs(
                    prompt, ground_truth
                )
            outputs = self.model(**encoded_input)
            probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
            target_idx = target_label[i]
            gt_prob = probs[:, target_idx].item()

            # get info about argmax completion
            argmax_prob, argmax_id = [i.item() for i in probs.max(dim=-1)]
            argmax_str = self.tokenizer.decode([argmax_id])

            prompt += argmax_str
            argmax_completion_str += argmax_str
        return argmax_completion_str,



    def n_layers(self):
        return len(self._get_transformer_layers())

    def intermediate_size(self):
        if self.model_type == "bert":
            return self.model.config.intermediate_size
        else:
            return self.model.config.hidden_size * 4

    def scaled_input(self, activations=None, steps=20, device="cpu", baseline_vector_path=None,
                     layer_idx=0, encoded_input=None, mask_idx=None):
        """
        Tiles activations along the batch dimension - gradually scaling them over
        `steps` steps from 0 to their original value over the batch dimensions.

        `activations`: torch.Tensor
        original activations
        `steps`: int
        number of steps to take
        """

        def get_baseline_vector(baseline_vector_path):
            with open(baseline_vector_path, 'r') as f:
                baseline_vector = json.load(f)
            return baseline_vector

        if baseline_vector_path is None:
            diff_activations = activations
        else:
            base_vector = torch.Tensor(get_baseline_vector(f'{baseline_vector_path}/layer{layer_idx}.json')).to(
                device)
            diff_activations = activations - base_vector
        tiled_activations = einops.repeat(diff_activations, "b d -> (r b) d", r=steps)

        out = (tiled_activations * torch.linspace(start=0, end=1, steps=steps).to(device)[:, None])
        return out

    def get_baseline_with_activations(
            self, encoded_input, layer_idx, mask_idx
    ):
        """
        Gets the baseline outputs and activations for the unmodified model at a given index.

        `encoded_input`: torch.Tensor
            the inputs to the model from self.tokenizer.encode_plus()
        `layer_idx`: int
            which transformer layer to access
        `mask_idx`: int
            the position at which to get the activations (TODO: rename? with autoregressive models there's no mask, so)
        """

        def get_activations(model, layer_idx, mask_idx):
            """
            This hook function should assign the intermediate activations at a given layer / mask idx
            to the 'self.baseline_activations' variable
            """

            def hook_fn(acts):
                # for i in range(12):
                #     x= acts[:,i,:]
                if mask_idx is not None:
                    self.baseline_activations = acts[:, mask_idx, :]
                else:
                    self.baseline_activations = acts[:, -1, :]

            return register_hook(
                model,
                layer_idx=layer_idx,
                f=hook_fn,
                transformer_layers_attr=self.transformer_layers_attr,
                ff_attrs=self.input_ff_attr,
            )

        handle = get_activations(self.model, layer_idx=layer_idx, mask_idx=mask_idx)
        baseline_outputs = self.model(**encoded_input)
        handle.remove()
        baseline_activations = self.baseline_activations
        self.baseline_activations = None
        return baseline_outputs, baseline_activations

    def get_scores(
            self,
            prompt: str,
            ground_truth: str,
            batch_size: int = 10,
            steps: int = 20,
            attribution_method: str = "integrated_grads",
            pbar: bool = True,
            baseline_vector_path=None,
            normalization=False,
            k_path=1,
    ):
        """
        Gets the attribution scores for a given prompt and ground truth.
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """
        scores = []
        # encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        for layer_idx in range(self.n_layers()):
            layer_scores = self.get_scores_for_layer(
                prompt,
                ground_truth,
                layer_idx=layer_idx,
                batch_size=batch_size,
                steps=steps,
                attribution_method=attribution_method,
                baseline_vector_path=baseline_vector_path,
                k_path=k_path,
            )
            layer_scores = layer_scores.to('cpu')
            scores.append(layer_scores)
        stacked_scores = torch.stack(scores)
        stacked_scores = stacked_scores.to('cuda:0')
        return stacked_scores


    def get_coarse_neurons(
            self,
            prompt: str,
            ground_truth: str,
            batch_size: int = 10,
            steps: int = 20,
            threshold: float = None,
            adaptive_threshold: float = None,
            percentile: float = None,
            attribution_method: str = "integrated_grads",
            pbar: bool = True,
            baseline_vector_path=None,
            normalization=False,
            k_path=1,
    ) -> List[List[int]]:
        attribution_scores = self.get_scores(
            prompt,
            ground_truth,
            batch_size=batch_size,
            steps=steps,
            pbar=pbar,
            attribution_method=attribution_method,
            baseline_vector_path=baseline_vector_path,
            normalization=normalization,
            k_path=k_path
        )

        assert (
                sum(e is not None for e in [threshold, adaptive_threshold, percentile]) == 1
        ), f"Provide one and only one of threshold / adaptive_threshold / percentile"
        if adaptive_threshold is not None:
            threshold = attribution_scores.max().item() * adaptive_threshold
        if threshold is not None:
            return torch.nonzero(attribution_scores > threshold).cpu().tolist()
        else:
            s = attribution_scores.flatten().detach().cpu().numpy()
            return (
                torch.nonzero(attribution_scores > np.percentile(s, percentile))
                .cpu()
                .tolist()
            )

    def get_refined_neurons(
            self,
            prompts: List[str],
            ground_truth: str,
            negative_examples: Optional[List[str]] = None,
            p: float = 0.5,
            batch_size: int = 10,
            steps: int = 20,
            coarse_adaptive_threshold: Optional[float] = 0.3,
            coarse_threshold: Optional[float] = None,
            coarse_percentile: Optional[float] = None,
            quiet=False,
            baseline_vector_path=None,
            normalization=False,
            k_path=1
    ) -> List[List[int]]:
        """
        """
        assert isinstance(
            prompts, list
        ), "Must provide a list of different prompts to get refined neurons"
        assert 0.0 <= p < 1.0, "p should be a float between 0 and 1"

        n_prompts = len(prompts)
        coarse_neurons = []
        for prompt in tqdm(
                prompts, desc="Getting coarse neurons for each prompt...", disable=quiet
        ):
            coarse_neurons.append(
                self.get_coarse_neurons(
                    prompt,
                    ground_truth,
                    batch_size=batch_size,
                    steps=steps,
                    adaptive_threshold=coarse_adaptive_threshold,
                    threshold=coarse_threshold,
                    percentile=coarse_percentile,
                    pbar=False,
                    baseline_vector_path=baseline_vector_path,
                    normalization=normalization,
                    k_path=k_path
                )
            )
        if negative_examples is not None:
            negative_neurons = []
            for negative_example in tqdm(
                    negative_examples,
                    desc="Getting coarse neurons for negative examples",
                    disable=quiet,
            ):
                negative_neurons.append(
                    self.get_coarse_neurons(
                        negative_example,
                        ground_truth,
                        batch_size=batch_size,
                        steps=steps,
                        adaptive_threshold=coarse_adaptive_threshold,
                        threshold=coarse_threshold,
                        percentile=coarse_percentile,
                        pbar=False,
                        baseline_vector_path=baseline_vector_path,
                        normalization=normalization
                    )
                )
        # 下面是二次过滤的代码
        if not quiet:
            total_coarse_neurons = sum([len(i) for i in coarse_neurons])
            print(f"\n{total_coarse_neurons} coarse neurons found - refining")
        t = n_prompts * p
        refined_neurons = []
        c = collections.Counter()
        for neurons in coarse_neurons:
            for n in neurons:
                c[tuple(n)] += 1

        for neuron, count in c.items():
            if count > t:
                refined_neurons.append(list(neuron))

        # filter out neurons that are in the negative examples
        if negative_examples is not None:
            for neuron in negative_neurons:
                if neuron in refined_neurons:
                    refined_neurons.remove(neuron)

        total_refined_neurons = len(refined_neurons)
        if not quiet:
            print(f"{total_refined_neurons} neurons remaining after refining")
        return refined_neurons

    def get_scores_for_layer(
            self,
            prompt: str,
            ground_truth: str,
            layer_idx: int,
            batch_size: int = 10,
            steps: int = 20,
            attribution_method: str = "integrated_grads",
            baseline_vector_path=None,
            k_path=1
    ):
        assert steps % batch_size == 0
        n_batches = steps // batch_size

        # First we take the unmodified model and use a hook to return the baseline intermediate activations at our chosen target layer
        encoded_input, mask_idx, target_label, prompt = self._prepare_inputs(
            prompt, ground_truth,
        )

        # for autoregressive models, we might want to generate > 1 token
        n_sampling_steps = len(target_label)

        integrated_grads = []

        for i in range(n_sampling_steps):
            if i > 0:
                # retokenize new inputs
                encoded_input, mask_idx, target_label, prompt = self._prepare_inputs(
                    prompt, ground_truth
                )
            (
                baseline_outputs,
                baseline_activations,
            ) = self.get_baseline_with_activations(
                encoded_input, layer_idx, mask_idx
            )
            if n_sampling_steps > 1:
                argmax_next_token = (
                    baseline_outputs.logits[:, mask_idx, :].argmax(dim=-1).item()
                )
                next_token_str = self.tokenizer.decode(argmax_next_token)

            scaled_weights = self.scaled_input(activations=baseline_activations, steps=steps,
                                               device=self.device,
                                               baseline_vector_path=baseline_vector_path,
                                               layer_idx=layer_idx, )
            scaled_weights.requires_grad_(True)

            integrated_grads_this_step = []  # to store the integrated gradients

            for batch_weights in scaled_weights.chunk(n_batches):
                inputs = {
                    "input_ids": einops.repeat(
                        encoded_input["input_ids"], "b d -> (r b) d", r=batch_size
                    ),
                    "attention_mask": einops.repeat(
                        encoded_input["attention_mask"],
                        "b d -> (r b) d",
                        r=batch_size,
                    ),
                }
                if self.model_type == "bert":
                    inputs["token_type_ids"] = einops.repeat(
                        encoded_input["token_type_ids"],
                        "b d -> (r b) d",
                        r=batch_size,
                    )

                # then patch the model to replace the activations with the scaled activations
                patch_ff_layer(  # mode =replace
                    self.model,
                    layer_idx=layer_idx,
                    mask_idx=mask_idx,
                    replacement_activations=batch_weights,
                    transformer_layers_attr=self.transformer_layers_attr,
                    ff_attrs=self.input_ff_attr,
                )

                # then forward through the model to get the logits
                outputs = self.model(**inputs)

                # then calculate the gradients for each step w/r/t the inputs
                probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
                target_idx = target_label[i]
                grad = torch.autograd.grad(
                    torch.unbind(probs[:, target_idx]), batch_weights
                )[0]
                grad = grad.sum(dim=0)
                integrated_grads_this_step.append(grad)

                unpatch_ff_layer(
                    self.model,
                    layer_idx=layer_idx,
                    transformer_layers_attr=self.transformer_layers_attr,
                    ff_attrs=self.input_ff_attr,
                )

            # then sum, and multiply by W-hat / m
            integrated_grads_this_step = torch.stack(
                integrated_grads_this_step, dim=0
            ).sum(dim=0)
            integrated_grads_this_step *= baseline_activations.squeeze(0) / steps
            integrated_grads.append(integrated_grads_this_step)

            if n_sampling_steps > 1:
                prompt += next_token_str
        integrated_grads = torch.stack(integrated_grads, dim=0).sum(dim=0) / len(
            integrated_grads
        )
        return integrated_grads

    def get_mastered_knowledge(self, prompt: str, ground_truth: str) -> str:
        _, _, argmax_completion_str, _ = self._generate(prompt, ground_truth)
        return argmax_completion_str

    def modify_activations(
            self,
            prompt: str,
            ground_truth: str,
            neurons: List[List[int]],
            mode: str = "suppress",
            undo_modification: bool = True,
            quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        results_dict = {}
        _, mask_idx, _ , prompt= self._prepare_inputs(
            prompt, ground_truth
        )  # just need to get the mask index for later - probably a better way to do this
        # get the baseline probabilities of the ground-truth being generated +
        # the argmax / greedy completion before modifying the activations
        (
            gt_baseline_prob,
            argmax_baseline_prob,
            argmax_completion_str,
            _,
        ) = self._generate(prompt, ground_truth)
        if not quiet:
            print(
                f"\nBefore modification - groundtruth probability: {gt_baseline_prob}\nArgmax completion:"
                f" `{argmax_completion_str}`\nArgmax prob: {argmax_baseline_prob}\n"
            )
        results_dict["before"] = {
            "gt_prob": gt_baseline_prob,
            "argmax_completion": argmax_completion_str,
            "argmax_prob": argmax_baseline_prob,
        }

        # patch model to suppress neurons
        # store all the layers we patch, so we can unpatch them later
        all_layers = set([n[0] for n in neurons])

        patch_ff_layer(
            self.model,
            mask_idx,
            mode=mode,
            neurons=neurons,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        # get the probabilities of the ground_truth being generated +
        # the argmax / greedy completion after modifying the activations
        new_gt_prob, new_argmax_prob, new_argmax_completion_str, _ = self._generate(
            prompt, ground_truth
        )
        if not quiet:
            print(
                f"\nAfter modification - groundtruth probability: {new_gt_prob}\nArgmax completion: "
                f"`{new_argmax_completion_str}`\nArgmax prob: {new_argmax_prob}\n"
            )
        results_dict["after"] = {
            "gt_prob": new_gt_prob,
            "argmax_completion": new_argmax_completion_str,
            "argmax_prob": new_argmax_prob,
        }

        unpatch_fn = partial(
            unpatch_ff_layers,
            model=self.model,
            layer_indices=all_layers,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        if undo_modification:
            unpatch_fn()
            unpatch_fn = lambda *args: args

        return results_dict, unpatch_fn


    def get_predict_acc(self, prompt, ground_truth, neurons, mode='suppress'):
        """原始的方法，为了防止错误被保留。实际上使用dkn里的方法"""
        _, mask_idx, _ , prompt= self._prepare_inputs(prompt, ground_truth)

        # Patch the model to suppress neurons
        patch_ff_layer(
            self.model,
            mask_idx,
            mode=mode,
            neurons=neurons,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        # Get the probability of the ground truth
        new_gt_prob, _, _, _ = self._generate(prompt, ground_truth)
        unpatch_fn = partial(
            unpatch_ff_layers,
            model=self.model,
            layer_indices=set([n[0] for n in neurons]),
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )
        unpatch_fn()
        unpatch_fn = lambda *args: args

        return new_gt_prob

    def get_acc_without_mode(self, prompt, ground_truth):
        encoded_input, mask_idx, target_label, prompt = self._prepare_inputs(
            prompt, ground_truth
        )
        n_sampling_steps = len(target_label)
        all_gt_probs = []
        for i in range(n_sampling_steps):
            if i > 0:
                encoded_input, mask_idx, target_label, prompt = self._prepare_inputs(
                    prompt, ground_truth
                )
            outputs = self.model(**encoded_input)
            probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
            target_idx = target_label[i]
            gt_prob = probs[:, target_idx].item()
            all_gt_probs.append(gt_prob)

        gt_prob = math.prod(all_gt_probs) if len(all_gt_probs) > 1 else all_gt_probs[0]
        return gt_prob



    def get_right_or_wrong(self, prompt, ground_truth, neurons, mode='suppress'):
        results_dict = {}
        _, mask_idx, _, prompt = self._prepare_inputs(
            prompt, ground_truth
        )

        all_layers = set([n[0] for n in neurons])

        patch_ff_layer(
            self.model,
            mask_idx,
            mode=mode,
            neurons=neurons,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        # get the probabilities of the ground_truth being generated +
        # the argmax / greedy completion after modifying the activations
        new_argmax_completion_str= self._get_answer_str(
            prompt, ground_truth
        )
        results_dict["after"] = {
            "argmax_completion": new_argmax_completion_str,
        }

        unpatch_fn = partial(
            unpatch_ff_layers,
            model=self.model,
            layer_indices=all_layers,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )
        unpatch_fn()
        # unpatch_fn = lambda *args: args
        return results_dict


    def suppress_knowledge(
            self,
            prompt: str,
            ground_truth: str,
            neurons: List[List[int]],
            undo_modification: bool = True,
            quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        prompt the model with `prompt`, zeroing the activations at the positions specified by `neurons`,
        and measure the resulting effect on the ground truth probability.
        """
        return self.modify_activations(
            prompt=prompt,
            ground_truth=ground_truth,
            neurons=neurons,
            mode="suppress",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    def enhance_knowledge(
            self,
            prompt: str,
            ground_truth: str,
            neurons: List[List[int]],
            undo_modification: bool = True,
            quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        prompt the model with `prompt`, multiplying the activations at the positions
        specified by `neurons` by 2, and measure the resulting affect on the ground truth probability.
        """
        return self.modify_activations(
            prompt=prompt,
            ground_truth=ground_truth,
            neurons=neurons,
            mode="enhance",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    @torch.no_grad()
    def modify_weights(
            self,
            prompt: str,
            neurons: List[List[int]],
            target: str,
            mode: str = "edit",
            erase_value: str = "zero",
            undo_modification: bool = True,
            quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        Update the *weights* of the neural net in the positions specified by `neurons`.
        Specifically, the weights of the second Linear layer in the ff are updated by adding or subtracting the value
        of the word embeddings for `target`.
        """
        assert mode in ["edit", "erase"]
        assert erase_value in ["zero", "unk"]
        results_dict = {}

        _, _, target_label, prompt = self._prepare_inputs(prompt, target)
        # get the baseline probabilities of the target being generated + the argmax / greedy completion before modifying the weights
        (
            gt_baseline_prob,
            argmax_baseline_prob,
            argmax_completion_str,
            argmax_tokens,
        ) = self._generate(prompt, target)
        if not quiet:
            print(
                f"\nBefore modification - groundtruth probability: {gt_baseline_prob}\nArgmax completion: `{argmax_completion_str}`\nArgmax prob: {argmax_baseline_prob}"
            )
        results_dict["before"] = {
            "gt_prob": gt_baseline_prob,
            "argmax_completion": argmax_completion_str,
            "argmax_prob": argmax_baseline_prob,
        }

        # get the word embedding values of the baseline + target predictions
        word_embeddings_weights = self._get_word_embeddings()
        if mode == "edit":
            assert (
                    self.model_type == "bert"
            ), "edit mode currently only working for bert models - TODO"
            original_prediction_id = argmax_tokens[0]
            original_prediction_embedding = word_embeddings_weights[
                original_prediction_id
            ]
            target_embedding = word_embeddings_weights[target_label]

        if erase_value == "zero":
            erase_value = 0
        else:
            assert self.model_type == "bert", "GPT models don't have an unk token"
            erase_value = word_embeddings_weights[self.unk_token]

        # modify the weights by subtracting the original prediction's word embedding
        # and adding the target embedding
        original_weight_values = []  # to reverse the action later
        for layer_idx, position in neurons:
            output_ff_weights = self._get_output_ff_layer(layer_idx)
            if self.model_type == "gpt":
                # since gpt2 uses a conv1d layer instead of a linear layer in the ff block, the weights are in a different format
                original_weight_values.append(
                    output_ff_weights[position, :].detach().clone()
                )
            else:
                # for BERT, BART encoder, and BART decoder, handle the weights this way.
                # BART follows the standard Transformer model, which uses Linear layers instead of Conv1D in its feed-forward blocks.
                original_weight_values.append(
                    output_ff_weights[:, position].detach().clone()
                )
            if mode == "edit":
                if self.model_type == "gpt":
                    output_ff_weights[position, :] -= original_prediction_embedding * 2
                    output_ff_weights[position, :] += target_embedding * 2
                else:
                    output_ff_weights[:, position] -= original_prediction_embedding * 2
                    output_ff_weights[:, position] += target_embedding * 2
            else:
                if self.model_type == "gpt":
                    output_ff_weights[position, :] = erase_value
                else:
                    output_ff_weights[:, position] = erase_value

        # get the probabilities of the target being generated + the argmax / greedy completion after modifying the weights
        (
            new_gt_prob,
            new_argmax_prob,
            new_argmax_completion_str,
            new_argmax_tokens,
        ) = self._generate(prompt, target)
        if not quiet:
            print(
                f"\nAfter modification - groundtruth probability: {new_gt_prob}\nArgmax completion: `{new_argmax_completion_str}`\nArgmax prob: {new_argmax_prob}"
            )
        results_dict["after"] = {
            "gt_prob": new_gt_prob,
            "argmax_completion": new_argmax_completion_str,
            "argmax_prob": new_argmax_prob,
        }

        def unpatch_fn():
            # reverse modified weights
            for idx, (layer_idx, position) in enumerate(neurons):
                output_ff_weights = self._get_output_ff_layer(layer_idx)
                if self.model_type == "gpt":
                    output_ff_weights[position, :] = original_weight_values[idx]
                else:
                    # for BERT, BART encoder, and BART decoder, handle the weights this way.
                    output_ff_weights[:, position] = original_weight_values[idx]

        if undo_modification:
            unpatch_fn()
            unpatch_fn = lambda *args: args

        return results_dict, unpatch_fn

    def edit_knowledge(
            self,
            prompt: str,
            target: str,
            neurons: List[List[int]],
            undo_modification: bool = True,
            quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        return self.modify_weights(
            prompt=prompt,
            neurons=neurons,
            target=target,
            mode="edit",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    def erase_knowledge(
            self,
            prompt: str,
            neurons: List[List[int]],
            erase_value: str = "zero",
            target: Optional[str] = None,
            undo_modification: bool = True,
            quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        return self.modify_weights(
            prompt=prompt,
            neurons=neurons,
            target=target,
            mode="erase",
            erase_value=erase_value,
            undo_modification=undo_modification,
            quiet=quiet,
        )

