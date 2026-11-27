---
name: xpl-mineral-assets
description: 为正交偏光薄片准备生成请求，打包透明矿物颗粒、导出整颗单矿物掩膜，或用指定光学模型渲染角度序列。
---

# 正交偏光矿物资产

从此技能目录运行脚本，所有请求、模型输出、资产和审核记录写入技能目录之外。运行环境与参考库布局见仓库根目录的 README。

1. 从完整请求提取矿物、物相、颗粒数量、所需现象、切面与成像方式。用户需要逼真的薄片外观时，使用 `scripts/xpl.py prepare --mode appearance-only --mineral NAME --description TEXT --out DIR`。各项特征可用重复的 `--phenomenon` 明确声明。
2. 用 `scripts/phenomena.py lookup TERM` 查现象，用 `scripts/atlas.py lookup NAME` 查已安装的本地资料。缺资料时标记证据缺口。检查与目标匹配、允许使用的真实照片，在图像模型支持时实际传入参考图。选中条目与实际传图分别记录。
3. 阅读生成的提示词。调用支持透明背景的图像模型，逐颗生成，保留实际返回的原始图像。在请求目录中记录实际使用的参考、工具和结果。检查晶界、完整轮廓、纹理和内部结构。
4. 用 `scripts/xpl.py build --request REQUEST_JSON --texture RGBA_IMAGE --out ASSET_DIR` 打包，再执行 `scripts/xpl.py validate --asset ASSET_DIR`。查看 `preview.html`，保留初始审核状态 `pending`。收到针对该资产的人工结论后，通过 `review` 记录意见。
5. 确认整个前景只属于一个矿物类别且透明边界可靠时，可用 `scripts/mineral_masks.py export --asset ASSET_DIR --out LABEL_DIR --class-id ID --single-mineral` 导出独立类别掩膜。双晶明暗和消光暗区仍属于同一矿物，内部相界需要独立证据。
6. 需要计算角度序列或光学参数场时，使用 `prepare --profile` 选取支持的模型，接着执行 `build`、`verify-angles` 和 `render`。描述厚度、折射率及取向假设。斜长石的图像引导双晶分区具有非唯一性。数值双晶演示只用于检验计算。程序检查通过后仍需人工检查外观。
