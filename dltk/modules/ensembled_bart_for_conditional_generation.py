# -*- encoding: utf-8 -*-
"""
@File    :   ensembled_bart_for_conditional_generation.py
@Time    :   2023/04/13 16:09:57
@Author  :   jiangjiajia
"""
from typing import Dict, Any, List
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.utils import ModelOutput
from transformers.models.bart.modeling_bart import *
from transformers.generation.utils import *

_CONFIG_FOR_DOC = "BartConfig"


class EnsembledBartForConditionalGeneration(BartPretrainedModel):
    def __init__(self, config_path, model_path_list, model_weight):
        config = AutoConfig.from_pretrained(config_path)
        super().__init__(config)
        self.models = torch.nn.ModuleList()
        for model_path in model_path_list:
            self.models.append(BartForConditionalGeneration.from_pretrained(model_path))
        self.models.eval()
        self.model_weight = model_weight

    def get_encoder(self):
        return [model.get_encoder() for model in self.models]

    def get_decoder(self):
        return [model.get_decoder() for model in self.models]

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[Tuple[Tuple[torch.FloatTensor]]]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if past_key_values is None:
            past_key_values = [past_key_values for _ in range(len(self.models))]

        outputs = []
        for model, each_model_encoder_outputs, each_model_past_key_value in zip(self.models, encoder_outputs, past_key_values):
            model.eval()
            outputs.append(model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=each_model_encoder_outputs,
                past_key_values=each_model_past_key_value,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ))
        loss = None
        if labels is not None:
            loss = 0.
            for each_outputs in outputs:
                loss += each_outputs.loss

        lm_logit_list = []
        for each_outputs in outputs:
            lm_logit_list.append(each_outputs.logits)
        # lm_logits /= len(self.models)

        if not return_dict:
            output = (lm_logit_list,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            # logits=torch.sum(torch.stack([o.logits * weight for o, weight in zip(outputs, self.model_weight)]), dim=0),
            logits=[o.logits for o in outputs],
            past_key_values=[o.past_key_values for o in outputs],
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=[o.encoder_last_hidden_state for o in outputs],
            encoder_hidden_states=None,
            encoder_attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        *args,
        **kwargs,
    ):
        return self.models[0].prepare_inputs_for_generation(*args, **kwargs)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past), )
        return reordered_past

    # @staticmethod
    # def _reorder_cache(past, beam_idx):
    #     reordered_past = ()
    #     for layer_past in past:
    #         # cached cross_attention states don't have to be reordered -> they are always the same
    #         reordered_past += (
    #             tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
    #         )
    #     return reordered_past

    def _prepare_encoder_decoder_kwargs_for_generation(
            self,
            inputs_tensor: torch.Tensor,
            model_kwargs,
            model_input_name: Optional[str] = None) -> Dict[str, Any]:
        # 1. get encoder
        encoders = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ['decoder_', 'cross_attn', 'use_cache']
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs['return_dict'] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs['encoder_outputs'] = [
            encoder(**encoder_kwargs) for encoder in encoders
        ]

        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: Optional[torch.LongTensor] = None,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(
                        dict_to_expand[key], torch.Tensor):
                    dict_to_expand[key] = dict_to_expand[
                        key].repeat_interleave(
                            expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            encoder_outputs = model_kwargs.get('encoder_outputs')
            if encoder_outputs is None:
                raise ValueError(
                    'If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.'
                )

            # Fix for ensemble
            for encoder_output in encoder_outputs:
                encoder_output[
                    'last_hidden_state'] = encoder_output.last_hidden_state.repeat_interleave(
                        expand_size, dim=0)
            model_kwargs['encoder_outputs'] = encoder_outputs

        return input_ids, model_kwargs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs['past_key_values'] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format)

        # update token_type_ids with last value
        if 'token_type_ids' in model_kwargs:
            token_type_ids = model_kwargs['token_type_ids']
            model_kwargs['token_type_ids'] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if 'attention_mask' in model_kwargs:
                attention_mask = model_kwargs['attention_mask']
                model_kwargs['attention_mask'] = torch.cat([
                    attention_mask,
                    attention_mask.new_ones((attention_mask.shape[0], 1))
                ],
                    dim=-1)
        else:
            # update decoder attention mask
            if 'decoder_attention_mask' in model_kwargs:
                decoder_attention_mask = model_kwargs['decoder_attention_mask']
                model_kwargs['decoder_attention_mask'] = torch.cat(
                    [
                        decoder_attention_mask,
                        decoder_attention_mask.new_ones(
                            (decoder_attention_mask.shape[0], 1))
                    ],
                    dim=-1,
                )

        return model_kwargs

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.
        Return:
            [`generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.BeamSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList(
        )
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList(
        )
        if max_length is not None:
            warnings.warn(
                '`max_length` is deprecated in this function, use'
                ' `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.',
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn(
                "You don't have defined any stopping_criteria, this will likely loop forever",
                UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else
            self.generation_config.output_attentions)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.generation_config.output_hidden_states)
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else
            self.generation_config.return_dict_in_generate)

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f'Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}.'
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if
            (return_dict_in_generate and output_scores) else None)
        decoder_attentions = () if (return_dict_in_generate
                                    and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate
                                  and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate
                                       and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs['encoder_outputs'].get(
                'attentions') if output_attentions else None
            encoder_hidden_states = (
                model_kwargs['encoder_outputs'].get('hidden_states')
                if output_hidden_states else None)

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams),
                                  dtype=torch.float,
                                  device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams, ))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need
            # # 算数平均
            # next_token_logits = torch.sum(torch.stack([o * weight for o, weight in zip(outputs.logits, self.model_weight)]), dim=0)[:, -1, :]
            # # next_token_logits = outputs.logits[:, -1, :]
            # # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            # next_token_logits = self.adjust_logits_during_generation(
            #     next_token_logits, cur_len=cur_len)
            # next_token_scores = nn.functional.log_softmax(
            #     next_token_logits,
            #     dim=-1)  # (batch_size * num_beams, vocab_size)

            # next_token_scores = []
            # for each_logits in outputs.logits:
            #     temp_next_token_logits = each_logits[:, -1, :]
            #     temp_next_token_logits = self.adjust_logits_during_generation(
            #         temp_next_token_logits, cur_len=cur_len)
            #     temp_next_token_scores = nn.functional.log_softmax(
            #         temp_next_token_logits,
            #         dim=-1)
            #     next_token_scores.append(temp_next_token_scores)
            # next_token_scores = torch.sum(torch.stack([scores * weight for scores, weight in zip(next_token_scores, self.model_weight)]), dim=0)

            # next_token_scores = []
            # for each_logits in outputs.logits:
            #     temp_next_token_logits = each_logits[:, -1, :]
            #     temp_next_token_logits = self.adjust_logits_during_generation(
            #         temp_next_token_logits, cur_len=cur_len)
            #     temp_next_token_scores = nn.functional.softmax(temp_next_token_logits, dim=-1)
            #     next_token_scores.append(temp_next_token_scores)
            # next_token_scores = torch.sum(torch.stack([scores * weight for scores, weight in zip(next_token_scores, self.model_weight)]), dim=0)
            # next_token_scores = torch.log(next_token_scores)


            next_token_scores = []
            for each_logits in outputs.logits:
                temp_next_token_logits = each_logits[:, -1, :]
                temp_next_token_logits = self.adjust_logits_during_generation(
                    temp_next_token_logits, cur_len=cur_len)
                temp_next_token_scores = nn.functional.softmax(temp_next_token_logits, dim=-1)
                next_token_scores.append(temp_next_token_scores)
            temp = torch.ones_like(next_token_scores[0])
            for o, weight in zip(next_token_scores, self.model_weight):
                temp = temp * (o * weight)
            next_token_scores = torch.log(torch.pow(temp, 1/len(next_token_scores)))


            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed, )
                if output_attentions:
                    decoder_attentions += ((outputs.decoder_attentions, ) if
                                           self.config.is_encoder_decoder else
                                           (outputs.attentions, ))
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions, )

                if output_hidden_states:
                    decoder_hidden_states += ((outputs.decoder_hidden_states, )
                                              if self.config.is_encoder_decoder
                                              else (outputs.hidden_states, ))

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size,
                                                       num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores,
                2 * num_beams,
                dim=1,
                largest=True,
                sorted=True)

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode='floor')
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs['next_beam_scores']
            beam_next_tokens = beam_outputs['next_beam_tokens']
            beam_idx = beam_outputs['next_beam_indices']

            input_ids = torch.cat(
                [input_ids[beam_idx, :],
                 beam_next_tokens.unsqueeze(-1)],
                dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder)
            if (model_kwargs['past_key_values'] is not None) and (not model_kwargs['past_key_values'][0] is None):
                for i in range(len(model_kwargs['past_key_values'])):
                    model_kwargs['past_key_values'][i] = self._reorder_cache(
                        model_kwargs['past_key_values'][i], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple(
                    (beam_indices[beam_idx[i]] + (beam_idx[i], )
                     for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs['sequence_scores'] = None

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs['sequences'],
                    sequences_scores=sequence_outputs['sequence_scores'],
                    scores=scores,
                    beam_indices=sequence_outputs['beam_indices'],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs['sequences'],
                    sequences_scores=sequence_outputs['sequence_scores'],
                    scores=scores,
                    beam_indices=sequence_outputs['beam_indices'],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs['sequences']
