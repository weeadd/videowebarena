#!/bin/bash

commit_message='newest'

# 设置默认提交信息
[ -z "$commit_message" ] && commit_message="update the newest"

# 配置Git凭证存储（按需启用）
git config --global credential.helper store

# 移除旧的远程仓库配置
git remote remove origin 2>/dev/null # 静默删除，避免报错

# 添加正确的远程仓库
git remote add origin https://github.com/weeadd/videowebarena.git

# 提交更改
git add .
git commit -m "$commit_message"

# 强制推送到新仓库（注意分支名称匹配）
git push --set-upstream origin main -f