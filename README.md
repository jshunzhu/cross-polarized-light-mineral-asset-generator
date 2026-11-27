# Cross-Polarized Light Mineral Asset Generator

用于生成、渲染和检查正交偏光（XPL）岩石薄片矿物颗粒的 Python 工具和代理技能。外观模式把矿物描述整理成生图提示词，再将带透明边界的模型输出打包为可预览、可校验的资产。物理模式根据给定的取向、厚度和折射率计算颗粒随偏光角度变化的图像。

## 快速开始

以下命令从仓库根目录运行。需要 Python 3.10 或更高版本、NumPy 和 Pillow。

```shell
cd skill/xpl-mineral-assets
python -m pip install -r scripts/requirements.txt
python -X utf8 scripts/xpl.py prepare --mode appearance-only --mineral Quartz --description "One isolated irregular quartz grain in XPL, low first-order grey, transparent margin." --out ../../work/request
```

打开生成的 `work/request/imagegen-prompt.txt`，用支持透明背景的图像模型生成颗粒，保存为 `work/grain.png`。也可使用自己有权处理的 RGBA 图像。可选的参考照片需在实际使用的图像工具中单独传入。Python 命令本身不调用远程生图服务。

```shell
python -X utf8 scripts/xpl.py build --request ../../work/request/request.json --texture ../../work/grain.png --out ../../work/asset
python -X utf8 scripts/xpl.py validate --asset ../../work/asset
```

用浏览器打开 `work/asset/preview.html` 查看原图。确认前景只包含同一种矿物后，可以另行导出整颗矿物的类别掩膜。以下示例由调用方把类别编号设为 1，默认前景边界为 alpha ≥ 128。

```shell
python -X utf8 scripts/mineral_masks.py export --asset ../../work/asset --out ../../work/labels --class-id 1 --single-mineral
python -X utf8 scripts/mineral_masks.py validate --bundle ../../work/labels
```

## 物理模式与本地参考资料

程序有石英均匀消光、石英波状消光、斜长石钠长石律双晶和铁铝榴石理想各向同性模型。物理渲染需要本地光谱表，部分模型还需要矿物光学参数或与指定现象匹配的参考库。将环境变量 `XPL_REFERENCE_DIR` 设为这些资料所在的绝对目录，其下可包含以下内容。

| 相对路径 | 用途与来源 |
| --- | --- |
| `optics/CIE_std_illum_D65.csv`、`optics/CIE_xyz_1931_2deg.csv` | 官方原始 CSV 用于光谱积分，程序核对字节校验和。[CIE D65](https://cie.co.at/datatable/cie-standard-illuminant-d65) · [CIE 1931 2° 标准观察者](https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer) |
| `optics/quartz-reference.json` | 石英模型读取数值字段 `n_ordinary` 和 `n_extraordinary`。本实现采用 600 nm 的名义值，材料文件应记录来源。[Crystran 石英材料说明](https://www.crystran.com/optical-materials/crystal-quartz-sio2/) |
| `optics/burri1967-geneva.json` | 斜长石模型读取 `source_url` 和 `rows`。每行使用 `an_percent`、`phi_a_deg`、`rho_a_deg`、`phi_b_deg`、`rho_b_deg`、`n_alpha`、`n_gamma`。角度遵循来源的坐标约定。[日内瓦大学光轴表](https://athena.unige.ch/athena/capitani/nieuwenkampdemo_OLD/feldsp.html) |
| `atlas/` | 可选的本地图集索引。对格式兼容且有权使用的 Markdown 原文运行 `python scripts/atlas.py build --source SOURCE.md --out INDEX_DIRECTORY` 可建立索引。 |
| `phenomena/`、`source/` | 可选的完整本地参考库，包括现象目录、类别卡、图片清单和溯源资料。完整库可用 `python scripts/phenomena.py validate` 校验。要求指定参考图的请求会检查图像与清单中的哈希。 |

只准备上述两张 CIE 表，即可运行铁铝榴石的理想模型。以下命令沿用快速开始中的透明颗粒输入，要求前景具有可见的透射纹理。

```shell
python -X utf8 scripts/xpl.py prepare --profile almandine-isotropic --out ../../work/physical-request
python -X utf8 scripts/xpl.py build --request ../../work/physical-request/request.json --texture ../../work/grain.png --out ../../work/physical-asset
python -X utf8 scripts/xpl.py render --asset ../../work/physical-asset --angles 0 22.5 45 90 --out ../../work/frames
```

这个理想模型在正交偏光下各角度均为黑色，透明边界和矿物掩膜仍可用于定位。石英和斜长石模型可产生随角度变化的强度。单张合成图推导的斜长石内部域属于估计结果，需结合预览与诊断人工复核。

## 检查与许可

```shell
python -X utf8 -m unittest discover -s scripts
```

基础测试在临时目录创建数值夹具。依赖外部光谱表或参考库的测试会报告跳过，配置相应资料后会一并执行。原始薄片图像、图集、模型输入输出和审核记录由使用者本地管理。源码按 [MIT 许可证](LICENSE) 发布。代理工作流见 [SKILL.md](skill/xpl-mineral-assets/SKILL.md)。
