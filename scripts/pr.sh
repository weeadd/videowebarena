#!/bin/bash

# 任务列表
input_text_ls=("test_classifieds" "test_gitlab" "test_map" "test_reddit" "test_shopping" "test_shopping_admin")
start_end_idx_ls=("0 200" "0 200" "0 200" "0 200" "0 200" "0 200")

# 设定最大同时运行的 screen 任务数
MAX_CONCURRENT_SCREENS=2
CONDA_ENV="vwa"  # 需要激活的 conda 环境
LOG_DIR="logs_2-12_v3_with_VideoGUIsummary"  # 存放日志的目录
mkdir -p "$LOG_DIR"

echo "任务总数: ${#input_text_ls[@]}"
echo "最大并行 screen 任务数: $MAX_CONCURRENT_SCREENS"
echo "使用的 conda 环境: $CONDA_ENV"
echo "日志存放目录: $LOG_DIR"

# 逐个执行 input_text_ls 里的任务
for task_idx in "${!input_text_ls[@]}"; do
    input_text=${input_text_ls[$task_idx]}
    start_end_idx=${start_end_idx_ls[$task_idx]}
    IFS=' ' read -ra ADDR <<< "$start_end_idx"
    start_idx=${ADDR[0]}
    end_idx=${ADDR[1]}

    echo "============================="
    echo "开始执行任务: $input_text (索引范围: $start_idx-$end_idx)"
    echo "============================="

    # 任务执行池
    running_sessions=()

    for ((idx = start_idx; idx < end_idx; idx++)); do
        session_name="${input_text:5}${idx}"
        log_file="$LOG_DIR/${session_name}.log"

        echo "==============================================="
        echo "🚀 启动任务: screen -dmS $session_name ..."
        
        # 确保日志文件存在
        touch "$log_file"

        # 启动任务，并让 screen 记录日志
        screen -L -dmS "$session_name" bash -c "conda activate $CONDA_ENV; . /Users/kevin/Documents/vwa/videowebarena/scripts/run_videowa.sh $input_text $idx >$log_file 2>&1; exit" &
        pid=$!
        running_sessions+=("$session_name:$pid")

        echo "✅ 任务 $session_name 已启动，日志文件: $log_file"
        echo "当前运行中的任务数: ${#running_sessions[@]}/$MAX_CONCURRENT_SCREENS"

        # 控制并行任务数量
        while (( ${#running_sessions[@]} >= MAX_CONCURRENT_SCREENS )); do
            echo "⏳ 达到并发上限 ($MAX_CONCURRENT_SCREENS)，等待任务完成..."
            
            for i in "${!running_sessions[@]}"; do
                session_name_pid="${running_sessions[$i]}"
                IFS=':' read -r session_name pid <<< "$session_name_pid"

                # 检查进程是否存活
                if ! kill -0 "$pid" 2>/dev/null; then
                    echo "✅ 任务 $session_name (PID: $pid) 已完成"
                    if [[ -f "$LOG_DIR/${session_name}.log" ]]; then
                        echo "📜 任务日志 (${session_name}.log) 已完成 "
                    else
                        echo "⚠️ 警告: 日志文件未找到 (${session_name}.log)"
                    fi
                    unset "running_sessions[$i]"
                    break
                fi
            done
            sleep 30  # 每30秒检查一次
        done
    done

    # 等待当前任务组的所有任务完成
    echo "⌛ 等待任务组 $input_text (索引范围: $start_idx-$end_idx) 的所有任务执行完成..."
    for session_name_pid in "${running_sessions[@]}"; do
        IFS=':' read -r session_name pid <<< "$session_name_pid"
        wait "$pid"
        echo "✅ 任务 $session_name (PID: $pid) 完成，日志如下："
        if [[ -f "$LOG_DIR/${session_name}.log" ]]; then
            cat "$LOG_DIR/${session_name}.log"
        else
            echo "⚠️ 警告: 日志文件未找到 (${session_name}.log)"
        fi
    done
    echo "🎉 任务组 $input_text (索引范围: $start_idx-$end_idx) 完成！"
done

echo "✅ 所有任务已完成！"
