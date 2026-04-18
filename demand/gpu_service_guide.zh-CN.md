
# GPU 服务使用指南

为了支持同学们完成课程项目，我们提供了分布在以下四台服务器上的 GPU 资源：

```bash
10.176.34.113
10.176.34.117
10.176.34.112
10.176.37.31

````

```bash
curl http://10.176.34.113:8080/list
curl http://10.176.34.117:8080/list
curl http://10.176.34.112:8080/list
curl http://10.176.37.31:8080/list

```

总共有 32 张 GPU。由于资源有限，请严格遵守以下规则，避免占用他人需要使用的资源：

1. 每位用户**同一时间只能占用一张 GPU**。我们会在单台服务器内强制执行这条规则。不同服务器之间不会联动限制，因此请大家自觉遵守。
2. **不用时立即释放资源。**
3. **禁止将服务器资源用于与课程实验无关的用途。**

下面是使用说明。`<server>` 表示以上四台主机中的任意一台。

---

## 1. 初始化你的账号

首次使用前，请先上传你的 SSH 公钥。如果你不熟悉 SSH 密钥，可以使用如下命令生成：

```bash
ssh-keygen
```

### 示例：

```bash
# Linux 或 macOS
curl -X POST http://<server>:8080/init \
  -H "Content-Type: application/json" \
  -d '{ "id": "23210240000", "ssh_pub_key": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJOQdZuj68sXVm4oj0jH7W1vNTMM9OdQMHNcU52tmb43 lizihan@LAPTOP-L90RO67M" }'

# Windows PowerShell（不是 cmd）
$pub = (Get-Content "C:\Users\lizihan\.ssh\id_mlclass.pub" -Raw).Trim()

Invoke-RestMethod -Method POST `
  -Uri "http://10.176.34.112:8080/init" `
  -ContentType "application/json" `
  -Body (@{
    id = "22307110243"
    ssh_pub_key = $pub
  } | ConvertTo-Json)

```

### 参数说明：

* `id`：你的学号
* `ssh_pub_key`：你的 SSH 公钥（**请完整复制公钥文件中的全部内容**）

每台服务器你只需要初始化一次。  
如果你上传错了 key，重新执行 `/init` 即可。  
如果你切换到另一台服务器，则必须重新初始化。

---

## 2. 查看可用 GPU

查看某台服务器 GPU 可用情况：

```bash
# Linux 或 macOS
curl http://10.176.34.117:8080/list

# Windows PowerShell
Invoke-RestMethod -Method GET `
  -Uri "http://<server>:8080/list"
```

### 返回示例：

```json
{
  "ok": true,
  "total_gpu_count": 8,
  "free_gpu_count": 5,
  "used_gpu_count": 3
}
```

多数情况下，你只需要关注 `free_gpu_count`。

---

## 3. 启动环境

如果有可用 GPU，你可以启动一个环境：

### 使用 GPU：

```bash
Invoke-RestMethod -Method POST `
  -Uri "http://10.176.34.117:8080/start" `
  -ContentType "application/json" `
  -Body '{"id":"22307110243","gpu":1}'

```

```bash
# test
Invoke-RestMethod -Method POST `
  -Uri "http://10.176.37.31:8080/start" `
  -ContentType "application/json" `
  -Body '{"id":"22307110243","gpu":1}'

```



### 不使用 GPU（仅访问文件等）：

```bash
curl -X POST http://<server>:8080/start \
  -H "Content-Type: application/json" \
  -d '{ "id": "23210240000", "gpu": 0 }'
```

### 参数说明：

* `id`：你的用户 ID
* `gpu`：

  * `1` → 申请 GPU
  * `0` → 不需要 GPU

---

### 返回信息

如果启动成功，你会收到：

#### 1. SSH 连接信息

```bash
$r = Invoke-RestMethod -Method GET `
  -Uri "http://10.176.34.112:8080/submit_status/80739cbdc629777caa34464d610e2498"

$r | Format-List *


```



```bash
ssh -o IdentitiesOnly=yes root@10.176.34.117 -p 60055 -i "C:\Users\lizihan\.ssh\id_mlclass"
```

```bash
#第二个
ssh -o IdentitiesOnly=yes root@10.176.34.112 -p 36155 -i "C:\Users\lizihan\.ssh\id_mlclass"
```


```bash
# test
ssh -o IdentitiesOnly=yes root@10.176.37.31 -p 32921 -i "C:\Users\lizihan\.ssh\id_mlclass"
```

* 你将以 **root** 用户登录
* `<your_private_key_path>` 应为你之前上传公钥对应的私钥路径
* **重要：请保存 `ssh_port`**，否则你将无法重新连接

---

#### 2. 用户端口

为了支持 agent 系统，我们额外暴露了 3 个端口：

| 容器端口 | 主机端口 |
| -------- | -------- |
| 8080     | 映射后返回 |
| 8081     | 映射后返回 |
| 8082     | 映射后返回 |

示例：

```json
{
  "ok": true,
  "user_id": "23210240000",
  "require_gpu": true,
  "gpu_id": 0,
  "ssh_port": 40221,
  "user_ports": [
    {"container_port": 8080, "host_port": 40222},
    {"container_port": 8081, "host_port": 40223},
    {"container_port": 8082, "host_port": 40224}
  ]
}
```

使用这些信息访问你的环境。

### Workspace
`/workspace` 是一个和你的 ID 绑定的持久化挂载目录，但**不会在不同服务器之间共享**。

---

## 4. 停止环境

使用结束后，**请立即释放资源**：

```bash
Invoke-RestMethod -Method POST `
  -Uri "http://10.176.34.117:8080/finish" `
  -ContentType "application/json" `
  -Body '{"id":"22307110243"}'

```

```bash
# test
Invoke-RestMethod -Method POST `
  -Uri "http://10.176.34.112:8080/finish" `
  -ContentType "application/json" `
  -Body '{"id":"22307110243"}'

```

⚠️ **你必须在使用结束后释放资源**

⚠️ **你必须在使用结束后释放资源**

⚠️ **你必须在使用结束后释放资源**

---

## 常见情况

### 1. 尚未初始化

先执行 `/init`，然后再重试 `/start`。

---

### 2. 没有可用 GPU

* 当前没有空闲 GPU
* 稍后再试
* 或者使用不带 GPU 的方式启动（`"gpu": 0`）

---

### 3. 环境已存在

* 你已经有一个运行中的环境
* 先用 `/finish` 停掉，再重新启动

---

### 4. GPU 意外被共享

如果你发现有其他进程在使用你的 GPU：

* 这些服务器可能与更高优先级的用户共享
* 你的 GPU 可能被抢占
* 建议做法：

  * 释放当前环境
  * 在另一台服务器上重新启动一个新环境

---

## 完整示例

```bash
# 1. 初始化
curl -X POST http://<server>:8080/init \
  -H "Content-Type: application/json" \
  -d '{"id":"23210240000","ssh_pub_key":"ssh-ed25519 AAAA... alice@laptop"}'

# 2. 查看资源
curl http://<server>:8080/list

# 3. 使用 GPU 启动
curl -X POST http://<server>:8080/start \
  -H "Content-Type: application/json" \
  -d '{"id":"23210240000","gpu":1}'

# 4. 使用 ssh_port 和 user_ports 连接

# 5. 停止环境
curl -X POST http://<server>:8080/finish \
  -H "Content-Type: application/json" \
  -d '{"id":"23210240000"}'
```
