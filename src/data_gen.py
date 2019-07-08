
"""
SAVE_DIR=Data/test/
python data_utils.py \
	--bsz_per_host=32 \
	--num_core_per_host=1 \
	--seq_len=512 \
	--reuse_len=256 \
	--input_glob=Data/test/test_num.txt \
	--save_dir=${SAVE_DIR} \
	--num_passes=20 \
	--bi_data=True \
	--mask_alpha=6 \
	--mask_beta=1 \
	--num_predict=85 \
	--from_raw_text=False



python data_utils.py \
	--bsz_per_host=32 \
	--num_core_per_host=1 \
	--seq_len=512 \
	--reuse_len=256 \
	--input_glob=Data/test/test_num.txt \
	--save_dir=${SAVE_DIR} \
	--num_passes=20 \
	--bi_data=True \
	--mask_alpha=6 \
	--mask_beta=1 \
	--num_predict=60 \
	--from_raw_text=True \
	--sp_path=""

"""



