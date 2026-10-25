# 个人账号可用的存储接口候选

核查日期：2026-10-25。以下是官方公开文档核查，不代表已经登录用户账号或完成真实上传下载。内置搜索服务本次不可用，使用官方页面的直接 HTTP 访问核验；没有依据无关搜索结果作结论。

所有项目共用私有恢复索引 [tivilou/project-persistence](https://github.com/tivilou/project-persistence)，每个项目使用 `projects/<project-id>/`。Q-Attention 已迁入 `projects/q-attention/`。共用索引不要求所有项目选同一存储商，代码和部署配方仍由各项目维护。

用户反馈阿里云盘不再向个人开发者开放，因此已移除“申请阿里云盘个人应用”这个接入前提。已有 Alipan 适配器仅保留为未实测的历史原型，不是当前选定后端。

| 候选 | 官方接入与能力 | 实际限制与判断 |
|---|---|---|
| **123 云盘 OpenAPI** | 官方概述明确：购买开发者权益包即可开通 API，无需等待审核；凭据通过站内信提供。支持文件管理、大文件断点续传/并行上传与下载。 | 优先做国内服务器候选。开发者权益与存储/下载会员不是同一项；下载接口列有非 VIP 自用下载每日1 GB上限。未核实当前售价和该账号权益，购买前需核对。 |
| **坚果云 WebDAV** | 在个人账号安全选项生成应用密码即可让程序上传、下载和管理文件，无需自建开发者应用。 | 接入简单。帮助页列出默认单文件500 M、免费每30分钟600请求/付费1500请求、单次列举750项且需分页。64 MiB备份块可适配文件大小限制；仍需核对套餐总容量和传输流量，不能仅凭支持WebDAV就承诺TB级恢复。 |
| **OneDrive / Microsoft Graph** | 上传会话 API 明确支持个人微软账号的 Files.ReadWrite 委托权限，支持按范围续传。 | 个人账号可用；需要应用注册和用户授权，不能将个人账号当作企业 app-only 服务账户。先实测所用 GPU 服务器的网络和吞吐。 |
| **Google Drive API** | 官方 Drive API 面向 My Drive 与 shared drives，提供文件上传、下载、查询和 OAuth 授权。 | 可用于个人 My Drive；优先选择最小文件权限。OAuth 测试/生产发布状态和 scope 审核要求需要按实际应用处理；不能把临时测试授权直接视作长期无人值守凭据。国内服务器先验证网络。 |
| **Dropbox API** | 官方开发者指南允许创建开发状态应用，仅自己使用可保持开发状态；有 App Folder 权限边界及文件 API。 | 适合作为海外候选；自行应用授权，长期调用需按官方 OAuth 管理凭据。购买容量前先测服务器连接和下载吞吐。 |
| **百度网盘开放平台** | 官方实名认证介绍仍列出个人开发者，可创建1个应用，具有上传、下载和文件管理能力。 | 认证说明较旧（2022）；较新的权限页（2026-07-03）规定未审核应用仅测试，10次/小时、10用户。正式配额需审核申请，不能作为即开即用的大规模恢复首选。默认在应用授权的 /apps/{appname} 范围操作。 |
| **pCloud API** | 官方提供 HTTP/JSON、OAuth、上传、文件链接与校验接口，账号数据按美国/欧洲机房选 API host。 | 技术上可作个人云盘候选；本次未核验账号开通费用、真实网络吞吐或大规模下载权益，优先级低于上述已匹配的选择。 |

用户预计长期总量 **100 GB–1 TB**。当前建议优先评估 **123 云盘 OpenAPI** 的开发者/API权益以及覆盖该规模恢复的下载权益；坚果云不作为此规模的默认首选，除非套餐容量和传输额度明确适合。如果已有 OneDrive 等付费账号，先做网络与传输实测再决定是否复用。选型前不购买、不迁移现有备份、不启用新后端。

## 一手来源

- 123 [概述](https://123yunpan.yuque.com/org-wiki-123yunpan-muaork/cr6ced/ppsuasz6rpioqbyt)：无审核开通、断点续传能力。
- 123 [接入流程](https://123yunpan.yuque.com/org-wiki-123yunpan-muaork/cr6ced/hpengmyg32blkbg8)：开发者权益包、站内信凭据。
- 123 [下载接口](https://123yunpan.yuque.com/org-wiki-123yunpan-muaork/cr6ced/fnf60phsushn8ip2)：download_info、非VIP自用下载额度错误。
- 123 [开发须知](https://123yunpan.yuque.com/org-wiki-123yunpan-muaork/cr6ced/txgcvbfgh0gtuad5)：令牌计入设备、同时至多3个，文档称有效期30天；实际到期以接口 expiredAt 为准。
- 123 [上传流程](https://123yunpan.yuque.com/org-wiki-123yunpan-muaork/cr6ced/il16qi0opiel4889)：分片 PUT、MD5校验及异步完成轮询。
- 坚果云 [WebDAV 开启及限制](https://help.jianguoyun.com/?p=2064)：个人应用密码、大小/频率/分页限制。
- 微软 [创建上传会话](https://learn.microsoft.com/en-us/graph/api/driveitem-createuploadsession?view=graph-rest-1.0)：个人账号权限、上传续传。
- Google [Drive API 概述](https://developers.google.com/workspace/drive/api/guides/about-sdk)、[OAuth scope 审核](https://developers.google.com/identity/protocols/oauth2/production-readiness/sensitive-scope-verification)。
- Dropbox [开发者指南](https://www.dropbox.com/developers/reference/developer-guide)：开发状态自用、生产审核与权限。
- 百度 [实名认证介绍](https://pan.baidu.com/union/doc/使用入门/实名认证/实名认证介绍/)、[权限与配额](https://pan.baidu.com/union/doc/使用入门/权限与配额/)。
- pCloud [官方 API](https://docs.pcloud.com/)、[OAuth](https://docs.pcloud.com/methods/oauth_2.0/authorize.html)。

123 新文档入口由官方站点 `yun.123pan.cn/devTools` 所引用的公开 DeveloperTools 前端资源链接发现，知识库ID42333972；旧 `www.123pan.com/union/doc` 在本次访问返回404，不能据此判断平台不开放。

## 接入实现边界

下一步才实现选定提供者的适配器。保持不可变快照、上传后回读、manifest最后提交和独立恢复验证；保持原实验 runner 的代码/数据/环境恢复校验。云端也使用每项目独立目录，但目录命名隔离不等于账号权限隔离。

现有工具的 Alipan backend 不可拿另一家 token 直接替换使用。接口、令牌生命周期、分页、分片及重试规则都需要对应实现和测试。当前没有任何新提供者的真实数据备份，也没有自动备份任务。
