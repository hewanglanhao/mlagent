

这是一个非常初步的 agent 实现。它没有在 agent 与工具之间做模块化拆分，并且各项功能的实现耦合度较高。该实现仅供参考。

# !!! 注意 !!!
现在你可以使用 `/submit-test` 来测试你的提交。更多细节见下文。

# 提交你的 Agent

请按照 `gpu_service_guide.md` 中的说明，将你的代码上传到 `10.176.37.31` 服务器。该服务器也可用于开发，但如果其他服务器仍有可用 GPU，请暂时不要使用这台服务器进行开发。




如果你已经在工作区中准备好了代码，可以使用 `/submit` 在评测容器中运行它。

## 提交前准备

### 1. 文件要求

请确保你的工作区包含：

* `run.sh`
* `run.sh` 所需的全部代码和文件

你的 `run.sh` 应位于：

```bash
/workspace/run.sh
```

一般来说，我们的评测环境已经包含常用依赖，包括 `openai` 以及相关性能分析工具。如果你还需要额外的 pip 包，可以在 `run.sh` 中使用以下命令安装。

```bash
pip3 install <your_package> -i --default-timeout 0.3 https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

你的 agent 应从以下位置读取评测目标规格：

```bash
/target/target_spec.json
```

你的 agent 应生成一个报告文件，并将其放置在：

```bash
/workspace/output.*
```

文件格式不受限制，但**只能**生成这样一个文件。

### 2. 模型/API 接口要求

你的 agent 应通过环境变量暴露模型接口，以便我们在评测时注入所使用的模型与 API 凭据。

可能会提供以下环境变量：

* `API_KEY`
* `BASE_MODEL`
* `BASE_URL`

如果你使用自己的 API key，可以不保留这些接口，但**不推荐**这样做。评测模型 `GPT-5.4` 已经是当前最强的代码模型之一。

推荐的代码风格如下：

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("API_KEY", ""),
    base_url=os.getenv("BASE_URL", "")
)

response = client.chat.completions.create(
    model=os.getenv("BASE_MODEL", ""),
    messages=[{"role": "user", "content": prompt}]
)
```

### 3. 规则

* 每位学生**同一时间最多只能有一个活跃任务**

  * 如果你已经有一个正在运行的 `/start` 环境，则不能再执行 `/submit`
  * 如果你已经有一个正在运行的 `/submit` 任务，则不能再次启动或提交
* 每位学生在 `4/21 8 am` 之前**最多可提交两次**
* 每次提交**最多运行 30 分钟**。超出该限制的提交可能会被自动终止。
* 有时你可能会遇到 API 限流或请求过于频繁的情况，稍等片刻后再重新提交即可。



## 提交

```bash
# linux 或 mac
curl -X POST http://<server>:8080/submit \
  -H "Content-Type: application/json" \
  -d '{ "id": "23210240000", "gpu": 1 }'
```

参数说明：

* `id`：你的学号
* `gpu`：
  * `1` 表示使用 GPU 提交（必需）


返回示例：

```json
{
  "ok": true,
  "user_id": "23210240000",
  "status": "running",
  "require_gpu": true,
  "gpu_id": 0,
  "output_file": "xxx",
  "submit_count": 0,
  "submit_limit": 2,
  "remaining_submit_count": 2
}
```

请保存返回的 `output_file`。
它是你查询提交状态时使用的标识符。

### 测试提交

这个提交入口仅用于测试。提交次数没有限制，并且使用的是 DeepSeek API。不过，它同样有整体 API token 限制，因此仍请节制使用。

```bash
# linux 或 mac
curl -X POST http://10.176.37.31:8080/submit-test \
  -H "Content-Type: application/json" \
  -d '{ "id": "22307110243", "gpu": 1 }'
```

```bash
# pwsh
$body = @{
  id  = "22307110243"
  gpu = 1
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://10.176.37.31:8080/submit-test" `
  -ContentType "application/json" `
  -Body $body

```

### 查询提交状态

```bash
curl http://10.176.37.31:8080/submit_status/<output_file>
```

可能的状态值包括：

* `running`
* `succeeded`
* `failed`
* `killed`

返回示例：

```json
{
  "ok": true,
  "output_file": "7f3d6d3b0d4f0b2f7a6d6d43b4b9fabc",
  "status": "succeeded",
  "gpu_id": 0,
  "started_at": 1713123456,
  "finished_at": 1713123510, // 如果未结束则为 null
  "submit_count": 1,
  "submit_limit": 2,
  "remaining_submit_count": 1
}
```

## 查看和下载输出

在浏览器中打开以下页面：

```text
http://10.176.37.31:8080/outputs
```

该页面会显示所有匿名输出文件的简单列表。
你可以点击任意链接直接下载对应文件。

使用你的输出 ID 下载你的输出文件。
或者你也可以通过启动容器在本地查看输出文件。

你的 agent 的标准输出和标准错误将写入：

```bash
/workspace/results.log
````

## 最小示例工作流

```bash
# 1. 上传你的代码到 /workspace
# 2. 确保 /workspace/run.sh 存在
# 3. 确保你的 agent 会读取 /target/target_spec.json
# 4. 提交

curl -X POST http://<server>:8080/submit \
  -H "Content-Type: application/json" \
  -d '{ "id": "23210240000", "gpu": 1 }'

# 5. 查询状态
curl http://<server>:8080/submit_status/<output_file>

# 6. 在浏览器中打开输出页面
# http://<server>:8080/outputs
```

# 提交你的报告

你需要提交一份简短报告，描述你构建的 agent，并附上一个由你的 agent 生成的、你认为较能代表其较好表现的 **output ID**。

```bash
/workspace/report.* # 文件格式不限
/workspace/output_id.txt # 你的 output ID
```
