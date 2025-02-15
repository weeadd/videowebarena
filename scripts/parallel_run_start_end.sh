#!/bin/bash  
#  input_text_ls=("test_classifieds" "test_gitlab" "test_map" "test_reddit" "test_shopping" "test_shopping_admin")  

# List of input texts  
  
input_text_ls=("test_classifieds")  
start_end_idx_ls=("0 999")  

  
for idx in "${!input_text_ls[@]}"; do  
    input_text=${input_text_ls[$idx]}  
    start_end_idx=${start_end_idx_ls[$idx]}  
    IFS=' ' read -ra ADDR <<< "$start_end_idx"  
    start_idx=${ADDR[0]}  
    end_idx=${ADDR[1]}  
    session_name=${input_text:5}$start_idx$end_idx 
    # echo $session_name
    screen -dmS $session_name bash -c ". scripts/run_videowa.sh $input_text $start_idx $end_idx; exit"  
done  