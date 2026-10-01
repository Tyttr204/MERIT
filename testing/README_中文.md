# System Test Runner 使用说明

本目录是现有统一部署的增量测试层。不要用它替换 main.py，不需要重新训练或校准。
将整个 testing 文件夹放到：

```text
<REPO_ROOT>\testing
```

所需层级：

```text
最终部署/
├── main.py                         # EAV-MAIN-INTEGRATION.1.0
├── system_config.json
├── system_config.eav.json
├── eeg/  audio/  video/  fusion/    # 已整理的原部署文件及资产
└── testing/
    ├── system_test_runner.py
    ├── test_scenarios.json          # 可选；使用时须显式传 --scenarios
    ├── run_regression.py
    ├── tests/test_system_test_runner.py
    └── development_test_report.json
```

只复制 system_test_runner.py 也可运行（可放在最终部署根目录或 testing 下）；它有内置场景默认值。不会导入训练、EQ1/EQ2/AQ系列实验脚本。读取和推理都使用原 main.SourceReader / main.Runtime.process_offline。

## 1. 先做不需要真实EAV和大模型的检查

```powershell
Set-Location -LiteralPath "<REPO_ROOT>"
$py = (Resolve-Path -LiteralPath "..\.venv-video\Scripts\python.exe").Path
& $py -X utf8 .\testing\system_test_runner.py --version
& $py -X utf8 .\testing\system_test_runner.py --self-test
& $py -X utf8 .\testing\system_test_runner.py --suite contracts --config .\system_config.json
```

- 版本：EAV-SYSTEM-TEST.1.0。
- 自检：38项NumPy逻辑和合成波形检查，不加载模型。
- contracts：实际加载原F4/AF4-B权重，人工概率和人工时钟。49项本脚本检查，并调用原融合模块52项preflight（检查有重叠，不把它们合计为独立实验数量）。
- contracts成功显示 PASS_CONTRACTS_ONLY，不是原始信号质量检测成功或情绪准确率。
- 包内不重复提供融合权重；读取已有配置所指的原资产。

## 2. 路径审计与计划检查

```powershell
& $py -X utf8 .\testing\system_test_runner.py --check-assets --config .\system_config.eav.json
& $py -X utf8 .\testing\system_test_runner.py --dry-run --suite reference --config .\system_config.eav.json
& $py -X utf8 .\testing\system_test_runner.py --dry-run --suite robustness --config .\system_config.eav.json
```

check-assets仅检查现有文件/脚本身份；模型格式、解码器和真实前向由实际运行检查。
dry-run沿用main的清单与路径检查，不解码波形，不加载情绪或质量模型。

默认选择：冻结六位VAL被试 × 五类 × 每类每人一个完整trial，得到30个trial、120个5秒窗口。
选样使用种子20260919和固定身份的哈希排序，不读取预测正确性、质量或confidence。
若只提供外部窗口JSONL，用 --selection all；需要EEG/Audio退化时，源清单仍必须包含对应trial的四个窗口。

原配置的EAV相对路径继续以system_config.eav.json所在目录为基准；脚本不寻找“最新实验目录”。
不要把输出目录命名为 test_*：原 main.py 的TEST源文件保护会拒绝该路径下的生成样本。
默认输出目录 system_checks 不触发此规则。

## 3. 真实单窗口preflight

仅在明确沿用并确认当前源EEG为微伏时使用下面的 uV；也可按真实单位提供 V/mV，或在原配置中固定单位。
程序不根据波形大小推测单位。既有记录 USER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED 不会因运行PASS而变成独立证明。

```powershell
& $py -X utf8 .\testing\system_test_runner.py --preflight --config .\system_config.eav.json --eeg-unit uV --window-key "subject10_instance010_w01"
```

这是一个真实窗口、一个reference条件、三个融合对照。真实加载并调用已有七个模块；不使用测试替身兜底。
没有 --window-key 时，preflight选择身份排序的第一个窗口并明确记录。
完整输入可因质量模块判断不可用而跳过某个分类器；软件调用成功不保证三模态全部可用。

## 4. 原始回放与缺失恢复

```powershell
# 一个窗口，reference + 七种非空缺失子集 + recovered，共9条案例
& $py -X utf8 .\testing\system_test_runner.py --suite smoke --config .\system_config.eav.json --eeg-unit uV --window-key "subject10_instance010_w01"

# 预先固定的均衡120窗口，只运行原始参考
& $py -X utf8 .\testing\system_test_runner.py --suite reference --config .\system_config.eav.json --eeg-unit uV
```

recovered重复原始输入，但重新运行模型，不读取之前预测。它是同载荷软件恢复检查，不是物理设备断开/重新连接测试。
恢复后不强制走F4，因为原参考本身不一定满足所有q>=0.80。
分类错误与软件失败分开：误分类不会让程序伪造PASS或失败；非法概率、错路由、错误缺失输入等硬约束才决定验收失败。

## 5. 真实信号退化测试

默认六类退化：

| 类型 | Mild / Medium / Severe |
|---|---|
| audio_attenuation | -6 / -18 / -30 dB |
| audio_white_noise | 原记录RMS/添加噪声RMS：20 / 10 / 0 dB |
| video_blur | sigma 1 / 2 / 4 |
| video_brightness | 亮度倍数 0.75 / 0.50 / 0.25 |
| eeg_global_line | 50Hz；参考/注入RMS：20 / 10 / 0 dB |
| eeg_channel_flatline | 3 / 9 / 18通道置0，目标集合嵌套 |

默认每窗口32个案例：原始参考1、Audio匹配参考1、Video匹配参考1、18个退化、EEG/Audio全零与Video全黑3、七种缺失7、恢复1。
120窗口即3840个案例。每案例最多重新执行三个模态；不要把3840写成独立被试数。

先缩小检查：

```powershell
# 先看计划：从均衡选择中截取一个完整trial，只用于调试
& $py -X utf8 .\testing\system_test_runner.py --dry-run --suite robustness --config .\system_config.eav.json --max-trials 1

# 四个5秒窗口，每个32案例，共128案例
& $py -X utf8 .\testing\system_test_runner.py --suite robustness --config .\system_config.eav.json --eeg-unit uV --max-trials 1

# 完整预选范围，显式使用可选场景文件
& $py -X utf8 .\testing\system_test_runner.py --suite robustness --config .\system_config.eav.json --scenarios .\testing\test_scenarios.json --eeg-unit uV
```

--max-trials按身份排序截取，明确标记BOUNDED_DEBUG_SUBSET，不是仍然代表六被试均衡集合。

还支持：

```text
eeg_local_broadband       六通道宽带噪声
eeg_global_broadband      三十通道独立宽带噪声
eeg_global_slow_drift     0.3Hz共享慢漂移
eeg_intermittent_hold     六通道，每5秒内冻结0.25/1/2.5秒
audio_pink_noise          合成1/f功率形状噪声
video_occlusion          图像中心矩形，约20%/40%/60%面积（不自动定位人脸）
video_compression        H264 CRF28/36/44，与CRF18匹配参考比较
```

使用 --families audio_attenuation,video_blur,eeg_global_line 指定子集，或 --families all 启用全部13类；all每窗口54案例。
默认不更改main配置或任何模型参数。严禁为让图表好看手改质量分或根据正确率选择样本。

### 匹配参考与剂量边界

1. EEG/Audio加噪使用完整20秒源trial确定RMS和随机噪声，然后切成四个5秒窗口。即使只推理其中一个窗口，计算剂量时也必须有原trial四个窗口；不是EEG/Audio全程都输入20秒模型。
2. Audio需要峰值余量时，为所选的Audio退化族及其匹配参考统一乘同一个增益；不得逐条件峰值归一化。原始reference保留不动，Audio趋势比较使用audio_matched_reference。所选族不同可能改变共同增益，报告中保留所选族与增益。
3. 原Audio窗口先由main按A0策略得到，再在测试副本中使用DOUBLE WAV无损保存数值，不额外引入PCM量化。
4. Video参考使用main的原5秒截取。滤镜测试增加RGB/FFV1匹配参考，减轻像素格式转换混杂；压缩单独使用CRF18参考。无需、也不修改原VQ1-B文件。
5. video_occlusion只遮挡图像中心，不能保证遮住所有真实人脸；因此不强制判定为不可用。video_blackout是全黑输入控制，必须无视觉证据；它不同于文件/数据流缺失。
6. 这些剂量是合成干扰探针，不是真实神经SNR、真实干净语音SNR，也不模拟所有故障。
7. 原数据、权重、校准器、源码只读；枚举的文件在运行前后检查SHA256。融合还会检查内存模型状态；不声称检查所有第三方依赖文件或所有情绪模型的内存状态。

## 6. 三种融合对照

- fixed_f4：对当前三头概率作固定F4融合。缺失输入同样先规范化，全部缺失时同样拒判。
- availability_only：对当前可用模态令q=1，缺失q=0；调用同一个冻结AF4-C。这是离线消融，不替换正式质量模块。
- quality_aware：main.py实际质量输出驱动的正式结果。

三者使用同一组当前情绪概率和同一组availability。不会为了基线再运行一套情绪模型，不修改分类器输出。
主程序的结果会与数值融合重算结果核对。q的阈值仍为0.80，AF4-B beta仍由原冻结资产提供。
权重解释只限自适应支路；F4没有伪造线性权重；全部缺失占位不是Neutral。

## 7. 输出与验收状态

默认：最终部署/system_checks/run_时间戳_套件名/。

| 文件 | 含义 |
|---|---|
| run_plan.json / source_selection.json | 推理前确定的场景、样本和种子 |
| run_config.json / environment.json | 本次配置与环境，不是升级建议 |
| loaded_asset_identity.json | 实际模块身份、加载耗时 |
| window_results.jsonl | 逐案例主程序结果、三基线、输入与检查 |
| case_manifest.jsonl | 每条场景的实际私有输入、源路径、变换和哈希 |
| comparison_predictions.csv | 三方案逐样本输出 |
| condition_metrics.csv / subject_metrics.csv / per_class_metrics.csv | 分组描述性统计 |
| quality_response_summary.csv | 质量单调性观察；只在四级都可用时统计连续质量趋势 |
| contract_checks.jsonl | 每一项硬性规则的通过/失败 |
| events.jsonl | 异常和完整traceback，不隐藏失败 |
| latency_summary.json | 模态支路和主流程耗时；比较开销单独列 |
| preparation_and_warmup.json | 原始读取准备、媒体转换与预热耗时 |
| media_commands.json | 实际视频变换命令及错误 |
| resource_usage.json | 进程RSS、PyTorch分配器峰值（不等于整块GPU占用） |
| protected_file_hashes.json / integrity_verification.json | 枚举资产/输入的内容一致性 |
| run_summary.json / comparison_summary.json / report.md | 总结 |

PASS_OFFLINE_PROCESSING只意味着预定范围处理及硬规则通过，不能宣称完整系统优于基线或已完成机器人真实部署。
COMPLETE_WITH_ERRORS/FAILED/FAILED_INTEGRITY/INTERRUPTED均不应作为成功验收。

主要性能汇总排除人为缺失、全零、全黑、恢复控制。各条件表保留控制记录用于单独审阅，不能把全缺失时accuracy=0当作系统失败。
同时记录作答覆盖率、已作答准确率、所有尝试准确率、软件错误与NO_DECISION；错误不是成功拒判。
Macro-F1固定五类；一个样本的preflight缺少其余类别，数值仅是格式/诊断信息，不是性能结论。
原参考不是认证完美输入。开发VAL已经用于选择质量特征及校准；没有新未见被试证据或统计显著性声明。

## 8. 错误、文件保留、预热、重新汇总

默认第一个错误停止，保留日志。调试时可显式 --continue-on-error；失败案例继续进入错误统计，最终不会报告PASS。内存不足仍停止。
没有自动resume，不覆盖既有结果。中断后可在新目录重新运行；旧结果可仅做统计：

```powershell
& $py -X utf8 .\testing\system_test_runner.py --summarize "这里填写某次system_checks运行目录"
```

此命令不重新加载模型、不读取原始数据，创建新报告目录，不修改旧日志。截断/损坏JSONL会明确报错，不静默丢尾行。

默认临时生成文件按trial清理，不长期保存敏感原始信号。需要演示媒体时，只对少量明确窗口使用 --retain-inputs；无损视频可能占用大量磁盘。
case_manifest中private_files_retained=false表示记录了输入身份，但对应临时文件已清理，不能把临时路径当作可播放资源。

--warmup 1 可在测量前额外运行一个真实参考窗口；默认0，日志明确标识第一个计时案例。不会用预热结果替代正式结果。
离线模型按main串行运行，耗时包含读取私有输入和情绪/质量计算，不含等待采集5秒，也不含额外基线比较。准备和模型加载另记。
仅FFmpeg子进程设置timeout；没有强杀原生/GPU模型调用的机制。Ctrl+C可能要等原生调用返回。没有自动启用摄像头、麦克风、EEG驱动或机械臂。

## 9. 开发回归测试

```powershell
& $py -X utf8 .\testing\run_regression.py --deployment-root . --output .\runner_regression_local.json
```

回归需要既有NumPy、SciPy、SoundFile、OpenCV、Torch与FFmpeg；不用安装新包。依赖缺失会明确失败，不伪造通过。
它使用真实F4/AF4-B与真实main读文件/调度路径，但情绪和质量模型为明确的测试替身，原始媒体为合成数据。
39项回归、38项自检和54场景管线测试，不等于对真实EAV的54场景性能测试。
发布前的真实验收必须由本机 --preflight / --suite reference / --suite robustness 完成。

## 10. 维护边界

本版不包含网页仪表盘、不运行额外模型、不重拟合q、不添加20秒质量聚合、不再训练AF4-C。
唯一用于原始输入推理的实现仍是main.py及现有七个模块。新输出字段为后续可视化提供数据。
只加载可信本地脚本与模型；路径配置和哈希检查不是对任意Python/checkpoint/joblib的安全沙箱。
