import os
import argparse
import json
import pandas as pd

import torch

from transformers import PreTrainedTokenizerFast
import tranception
from tranception import config, model_pytorch

dir_path = os.path.dirname(os.path.abspath(__file__))


def get_parser():
    parser = argparse.ArgumentParser(description='Tranception scoring')
    parser.add_argument('--checkpoint', type=str, help='Path of Tranception model checkpoint')
    # parser.add_argument('--model_framework', default='pytorch', type=str, help='Underlying framework [pytorch|JAX]')
    parser.add_argument('--batch_size_inference', default=20, type=int, help='Batch size for inference')
    parser.add_argument("--tokenizer_path",
                        default=os.path.join(dir_path, "tranception/utils/tokenizers/Basic_tokenizer"))

    # We may pass in all required information about the DMS via the provided reference files, or specify all relevant fields manually
    parser.add_argument('--DMS_reference_file_path', default=None, type=str,
                        help='Path to reference file with list of DMS to score')
    parser.add_argument('--DMS_index', default=0, type=int, help='Index of DMS assay in reference file')
    # Fields to be passed manually if reference file is not used
    parser.add_argument('--target_seq', default=None, type=str,
                        help='Full wild type sequence that is mutated in the DMS asssay')
    parser.add_argument('--DMS_file_name', default=None, type=str, help='Name of DMS assay file')
    parser.add_argument('--mutant_column', default=None, type=str,
                        help='Column name for mutant column in sequence file; note that in indel mode mutants have to be the full sequence')
    parser.add_argument('--DMS_data_folder', type=str, help='Path to folder that contains all DMS assay datasets')
    parser.add_argument('--output_scores_folder', default='./', type=str,
                        help='Name of folder to write model scores to')
    parser.add_argument('--indel_mode', action='store_true',
                        help='Flag to be used when scoring insertions and deletions. Otherwise assumes substitutions')
    # MSA args: If using retrieval
    parser.add_argument('--MSA_filename', default=None, type=str,
                        help='Name of MSA (eg., a2m) file constructed on the wild type sequence')
    parser.add_argument('--MSA_weight_file_name', default=None, type=str,
                        help='Weight of sequences in the MSA (optional)')
    parser.add_argument('--MSA_start', default=None, type=int,
                        help='Sequence position that the MSA starts at (1-indexing)')
    parser.add_argument('--MSA_end', default=None, type=int, help='Sequence position that the MSA ends at (1-indexing)')
    parser.add_argument('--inference_time_retrieval', action='store_true',
                        help='Whether to perform inference-time retrieval')
    parser.add_argument('--retrieval_inference_weight', default=0.6, type=float,
                        help='Coefficient (alpha) used when aggregating autoregressive transformer and retrieval')
    parser.add_argument('--MSA_folder', default='.', type=str, help='Path to MSA for neighborhood scoring')
    parser.add_argument('--MSA_weights_folder', default=None, type=str,
                        help='Path to MSA weights for neighborhood scoring')
    parser.add_argument('--clustal_omega_location', default=None, type=str,
                        help='Path to Clustal Omega (only needed with scoring indels with retrieval)')
    # Other arguments
    parser.add_argument('--deactivate_scoring_mirror', action='store_true',
                        help='Whether to deactivate sequence scoring from both directions (Left->Right and Right->Left)')
    parser.add_argument('--scoring_window', default="optimal", type=str,
                        help='Sequence window selection mode (when sequence length longer than model context size)')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of workers for model scoring data loader')
    return parser


def load_model(checkpoint,
               target_seq=None,
               indel_mode=False,
               inference_time_retrieval=False,
               retrieval_inference_weight=0.6,
               MSA_file_fullpath=None,
               MSA_weights_fullpath=None,
               MSA_start=None,
               MSA_end=None,
               clustal_omega_location=None,
               scoring_window="optimal",
               tokenizer_path=os.path.join(dir_path, "tranception/utils/tokenizers/Basic_tokenizer"),
               ):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )
    config = json.load(open(checkpoint + os.sep + 'config.json'))
    config = tranception.config.TranceptionConfig(**config)
    config.attention_mode = "tranception"
    config.position_embedding = "grouped_alibi"
    config.tokenizer = tokenizer
    config.scoring_window = scoring_window

    if inference_time_retrieval:
        config.retrieval_aggregation_mode = "aggregate_indel" if indel_mode else "aggregate_substitution"
        config.MSA_filename = MSA_file_fullpath
        config.full_protein_length = len(target_seq)
        config.MSA_weight_file_name = MSA_weights_fullpath
        config.retrieval_inference_weight = retrieval_inference_weight
        config.MSA_start = MSA_start
        config.MSA_end = MSA_end
        if indel_mode:
            config.clustal_omega_location = clustal_omega_location
    else:
        config.retrieval_aggregation_mode = None

    # if model_framework=="pytorch":
    model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(
        pretrained_model_name_or_path=checkpoint, config=config)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    return model


def _score_and_save_csv(model,
                        model_name,
                        batch_size_inference,
                        sequences_file,
                        DMS_id,
                        output_scores_folder,
                        mutant_column,
                        target_seq=None,
                        indel_mode=False,
                        inference_time_retrieval=0.6,
                        retrieval_inference_weight=0.6,
                        deactivate_scoring_mirror=False,  # Other stuff
                        num_workers=1,
                        ):
    if not os.path.isdir(output_scores_folder):
        os.mkdir(output_scores_folder)
    retrieval_type = '_retrieval_' + str(retrieval_inference_weight) if inference_time_retrieval else '_no_retrieval'
    mutation_type = '_indels' if indel_mode else '_substitutions'
    mirror_type = '_no_mirror' if deactivate_scoring_mirror else ''
    scoring_filename = output_scores_folder + os.sep + model_name + retrieval_type + mirror_type + mutation_type
    scoring_filename += os.sep + DMS_id + '.csv'
    if not os.path.isdir(scoring_filename):
        os.mkdir(scoring_filename)

    DMS_data = pd.read_csv(sequences_file, low_memory=False)

    if mutant_column is not None:
        # it's simpler to just rename the mutant column to 'mutant' than to change all the internal code
        assert mutant_column in DMS_data.columns, DMS_data.columns
        expected_mutant_column = 'mutant' if not indel_mode else 'mutated_sequence'

        if mutant_column != expected_mutant_column:
            # If both 'mutant' and our preferred mutant_column exist, keep the original mutant as mutant_old
            if expected_mutant_column in DMS_data.columns:
                DMS_data = DMS_data.rename(columns={expected_mutant_column: f"{expected_mutant_column}_old"})
            DMS_data = DMS_data.rename(columns={mutant_column: expected_mutant_column})

    all_scores = model.score_mutants(
        DMS_data=DMS_data,
        target_seq=target_seq,
        scoring_mirror=not deactivate_scoring_mirror,
        batch_size_inference=batch_size_inference,
        num_workers=num_workers,
        indel_mode=indel_mode
    )
    all_scores.to_csv(scoring_filename, index=False)


def read_mapping_file(DMS_reference_file_path, DMS_index, inference_time_retrieval, MSA_folder, MSA_weights_folder):
    MSA_kwargs = {}
    # Reading DMS reference file
    mapping_protein_seq_DMS = pd.read_csv(DMS_reference_file_path)
    list_DMS = mapping_protein_seq_DMS["DMS_id"]
    DMS_id = list_DMS[DMS_index]
    print("Compute scores for DMS: " + str(DMS_id))
    target_seq = mapping_protein_seq_DMS["target_seq"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[
        0].upper()
    DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0]
    if inference_time_retrieval:
        MSA_file_fullpath = MSA_folder + os.sep + mapping_protein_seq_DMS["MSA_filename"][
            DMS_index] if MSA_folder is not None else None
        MSA_weights_fullpath = MSA_weights_folder + os.sep + mapping_protein_seq_DMS["weight_file_name"][
            mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0] if MSA_weights_folder else None
        MSA_start = int(mapping_protein_seq_DMS["MSA_start"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[
                            0]) - 1  # MSA_start typically based on 1-indexing
        MSA_end = int(mapping_protein_seq_DMS["MSA_end"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0])
        MSA_kwargs = {
            "MSA_file_fullpath": MSA_file_fullpath,
            "MSA_weights_fullpath": MSA_weights_fullpath,
            "MSA_start": MSA_start,
            "MSA_end": MSA_end,
        }

    return target_seq, DMS_file_name, DMS_id, MSA_kwargs


def main():
    """
    Score a CSV file with Tranception (eventually modify to run with fasta)
    """
    parser = get_parser()
    args = parser.parse_args()

    MSA_kwargs = {}

    if args.DMS_reference_file_path:
        target_seq, DMS_file_name, DMS_id, MSA_kwargs = read_mapping_file(
            DMS_reference_file_path=args.DMS_reference_file_path,
            DMS_index=args.DMS_index,
            inference_time_retrieval=args.inference_time_retrieval,
            MSA_folder=args.MSA_folder,
            MSA_weights_folder=args.MSA_weights_folder,
        )
    else:
        target_seq = args.target_seq
        DMS_file_name = args.DMS_file_name
        DMS_basename = os.path.basename(DMS_file_name)
        DMS_id = DMS_basename.split(".")[0]
        if args.inference_time_retrieval:
            MSA_file_fullpath = os.path.join(args.MSA_folder,
                                             args.MSA_filename) if args.MSA_folder is not None else None
            MSA_weights_fullpath = os.path.join(args.MSA_weights_folder,
                                                args.MSA_weight_file_name) if args.MSA_weights_folder is not None else None
            MSA_start = args.MSA_start - 1  # MSA_start based on 1-indexing
            MSA_end = args.MSA_end
            MSA_kwargs = {
                "MSA_file_fullpath": MSA_file_fullpath,
                "MSA_weights_fullpath": MSA_weights_fullpath,
                "MSA_start": MSA_start,
                "MSA_end": MSA_end,
            }

    model_name = args.checkpoint.split("/")[-1]

    model = load_model(
        checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer_path,
        scoring_window=args.scoring_window,
        **MSA_kwargs,
    )

    sequence_file = args.DMS_data_folder + os.sep + DMS_file_name
    _score_and_save_csv(model=model,
                        sequences_file=sequence_file,
                        model_name=model_name,
                        DMS_id=DMS_id,
                        output_scores_folder=args.output_scores_folder,
                        batch_size_inference=args.batch_size_inference,
                        mutant_column=args.mutant_column,
                        indel_mode=args.indel_mode,
                        target_seq=target_seq,
                        inference_time_retrieval=args.inference_time_retrieval,
                        retrieval_inference_weight=args.retrieval_inference_weight,
                        deactivate_scoring_mirror=args.deactivate_scoring_mirror,
                        num_workers=args.num_workers,
                        )


if __name__ == '__main__':
    main()
