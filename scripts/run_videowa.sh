export max_frame_num=5
export videoframe_model="deepseek-v3"
export videoframe_model_provider="qwen"
export video_model="deepseek-v3"
export video_model_provider="qwen"
export test_config_dir="/Users/kevin/Documents/vwa/videowebarena/config_files/videowa"

export domain=$1 # test_classifieds
# export test_start_idx=$2 # 100
# export test_end_idx=$3 # 206
export test_idx_ls=$2
export test_config_base_dir="$test_config_dir/$domain"
export result_dir="/Users/kevin/Documents/vwa/videowebarena/results_2-12_v3_with_VideoGUIsummary/$DATASET/$domain/$test_start_idx-$test_end_idx" 
export video_domain=${domain:5}

export DATASET="videowebarena"
export CLASSIFIEDS="http://ec2-18-118-136-233.us-east-2.compute.amazonaws.com:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c" 
export SHOPPING="http://ec2-18-118-136-233.us-east-2.compute.amazonaws.com:7770"
export REDDIT="http://ec2-18-118-136-233.us-east-2.compute.amazonaws.com:9999"
export WIKIPEDIA="http://ec2-18-118-136-233.us-east-2.compute.amazonaws.com:8888"
export HOMEPAGE="http://ec2-18-118-136-233.us-east-2.compute.amazonaws.com:4399"
export SHOPPING_ADMIN="http://ec2-18-118-136-233.us-east-2.compute.amazonaws.com:7780/admin"
export GITLAB="http://ec2-18-118-136-233.us-east-2.compute.amazonaws.com:8023"
export MAP="http://ec2-18-118-136-233.us-east-2.compute.amazonaws.com:3000"
export QWEN_KEY="sk-1a962d482b194db3bd84b1b23caaf77d"



# video summary 
# rm -rf $result_dir
cd /Users/kevin/Documents/vwa/videowebarena/
python /Users/kevin/Documents/vwa/videowebarena/run.py \
  --instruction_path /Users/kevin/Documents/vwa/videowebarena/agent/prompts/jsons/p_som_cot_id_actree_3s_video_summary.json \
  --video_summary_instruction_path /Users/kevin/Documents/vwa/videowebarena/agent/prompts/jsons/video_frame_understanding.json \
  --test_idx_ls=$test_idx_ls\
  --test_config_base_dir=$test_config_base_dir \
  --provider=$video_model_provider \
  --model=$video_model\
  --action_set_tag som \
  --observation_type image_som\
  --result_dir $result_dir\
  --agent_type video_summary_prompt\
  --video_dir media\
  --max_frame_num 5 \
  --max_tokens 8000\
  --intermediate_intent_instruction_path /Users/kevin/Documents/vwa/videowebarena/agent/prompts/jsons/video_frame_intent_understanding.json



# video frame prompt with intermediate eval
# rm -rf $result_dir
# python run.py \
#   --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s_video_frame.json \
#   --test_start_idx=$test_start_idx \
#   --test_end_idx=$test_end_idx \
#   --test_config_base_dir=$test_config_base_dir \
#   --provider=$videoframe_model_provider \
#   --model=$videoframe_model\
#   --action_set_tag som \
#   --observation_type image_som\
#   --result_dir $result_dir\
#   --agent_type video_prompt\
#   --video_dir media\
#   --max_frame_num=$max_frame_num\
#   --max_tokens 4096\
#   --intermediate_intent_instruction_path agent/prompts/jsons/video_frame_intent_understanding.json

# video frame summary
# rm -rf $result_dir
# python run.py \
#   --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s_video_summary.json \
#   --video_summary_instruction_path agent/prompts/jsons/video_frame_understanding.json \
#   --test_start_idx=$test_start_idx \
#   --test_end_idx=$test_end_idx \
#   --test_config_base_dir=$test_config_base_dir \
#   --provider=$videoframe_model_provider \
#   --model=$videoframe_model\
#   --action_set_tag som \
#   --observation_type image_som\
#   --result_dir $result_dir\
#   --agent_type video_summary_prompt\
#   --video_dir media\
#   --max_tokens 4096\
#   --max_frame_num $max_frame_num\
#   --intermediate_intent_instruction_path agent/prompts/jsons/video_frame_intent_understanding.json


# video prompt with intermediate eval
# rm -rf $result_dir
# python run.py \
#   --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s_video.json \
#   --test_start_idx=$test_start_idx \
#   --test_end_idx=$test_end_idx \
#   --test_config_base_dir=$test_config_base_dir \
#   --provider=$video_model_provider \
#   --model=$video_model\
#   --action_set_tag som \
#   --observation_type image_som\
#   --result_dir $result_dir\
#   --agent_type video_prompt\
#   --video_dir media\
#   --mode completion\
#   --max_tokens 8000\
#   --intermediate_intent_instruction_path agent/prompts/jsons/video_intent_understanding.json









