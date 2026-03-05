# main.py
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from openai import AsyncOpenAI
from typing import Optional
import openai

# ==================== 配置区 ====================
MAX_RESUME_LENGTH = 10000      # 简历最大字符数
MAX_JD_LENGTH = 5000           # 岗位描述最大字符数
MAX_API_KEY_LENGTH = 100       # API Key 长度上限
# ================================================

app = FastAPI(title="安全简历修改大师 API")

# 允许跨域（开发阶段允许所有，生产环境请限制）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 请求数据模型 ====================
class TailorRequest(BaseModel):
    resume_text: str = Field(..., max_length=MAX_RESUME_LENGTH, description="原始简历文本")
    job_description: str = Field(..., max_length=MAX_JD_LENGTH, description="岗位描述文本")
    api_key: str = Field(..., max_length=MAX_API_KEY_LENGTH, description="用户的 DeepSeek API Key")
    company_name: Optional[str] = Field(None, max_length=200, description="目标公司名称（可选）")  # 注意这里是 None
   
    @validator("resume_text", "job_description")
    def sanitize_text(cls, v):
        # 移除控制字符，防止注入
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", v)

    @validator("api_key")
    def validate_api_key_format(cls, v):
        if not v.startswith("sk-"):
            raise ValueError("API Key 格式不正确，应以 sk- 开头")
        return v

# ==================== 根路径 ====================
@app.get("/")
async def root():
    return {"message": "简历修改大师 API 已启动，请访问 /docs 查看文档"}

# ==================== 核心接口 ====================
@app.post("/tailor")
async def tailor_resume(request: TailorRequest):
    print("=== 函数开始执行 ===", flush=True)
    print(f"收到请求: 简历长度={len(request.resume_text)}, JD长度={len(request.job_description)}, company_name={request.company_name}", flush=True)
    """
    接收用户简历、岗位描述和 API 密钥，调用 DeepSeek 优化简历。
    """
    # 1. 使用用户提供的 API 密钥创建异步客户端
    try:
        client = AsyncOpenAI(
            api_key=request.api_key,
            base_url="https://api.deepseek.com/v1",
            timeout=60.0
        )
    except Exception:
        print(f"客户端创建失败: {e}", flush=True)
        raise HTTPException(status_code=400, detail="无效的 API Key 格式")

    # 2. 构造提示词
    prompt = f"""
你是一位资深人力资源专家，拥有多年筛选简历的经验，深知名企HR的喜好以及如何通过ATS（申请跟踪系统）筛选。同时，你擅长在简历优化之外，帮助求职者深入了解目标公司，并提供针对性的面试指导，从而提高面试通过率。

现在，你的任务是根据用户提供的职位描述（JD）以及其原始的工作/实习经历、项目经验和技能清单，生成以下三部分内容：

公司分析：基于JD（如果用户提供了公司名称，则结合公开知识）对目标公司进行深度剖析，帮助学生理解公司背景和岗位定位。

简历内容优化：对用户的工作/实习经历、项目经验和技能清单进行针对性优化，使其与JD高度匹配，能够迅速抓住HR注意力并通过AI筛选。注意用户主要为在校学生，经历多为实习，优化时需确保描述符合实习生的实际工作范围和权限，不夸大，但通过STAR法则和量化突出真实贡献。

面试指导：针对优化后的简历，结合公司分析，为用户提供面试准备建议和实用技巧，帮助学生自信应对面试。

第一部分：公司分析要求
根据用户提供的JD（以及可能隐含的公司名称），对目标公司进行全面分析，帮助用户理解公司背景和岗位定位。分析应包含但不限于以下维度：

所属行业及细分领域：明确公司所处的行业赛道（如互联网、金融、消费、医疗等）和具体细分领域。

公司在行业内的地位：判断公司是行业头部、腰部玩家，还是新兴创业公司？其市场口碑、竞争优势如何？

公司当前的战略重点和业务发展方向：从JD中提取线索，推断公司为什么招聘这个岗位（如业务扩张、新项目启动、团队补位等），以及公司未来可能关注的重点方向。

融资阶段与规模：如果可从JD或常识推断，注明公司融资阶段（如A轮、B轮、上市公司）及大致人员规模。

岗位在公司的定位：该岗位是核心业务部门还是支持部门？汇报对象可能是谁？岗位面临的挑战和机会点是什么？

公司文化与价值观（如可推断）：从JD的语言风格、福利描述等推测公司文化（如扁平化、结果导向、创新驱动等）。

注：如果用户提供了公司名称，你可以结合自身知识库进行更具体的分析；如果未提供公司名称，请根据JD中透露的行业信息、岗位职责等进行合理推断，但避免虚构具体公司名称。

第二部分：简历内容优化要求
基于用户提供的原始经历和技能，针对JD进行深度优化。用户为在校学生，经历以实习为主，优化时需注意描述的真实性和合理性，不超出实习生的职责范围，但通过STAR法则和量化突出个人贡献与学习能力。 具体标准如下：

深度解析JD：提取JD中的关键词、技能要求、经验要求，并自然融入优化后的经历描述中，以提高ATS匹配度。

STAR法则重构经历：使用STAR法则（情境、任务、行动、结果）重新组织工作/实习经历和项目经验，强调具体行动和可量化的成果。对于实习经历，可适度描述在团队中的协作、支持角色，以及取得的实际成效。

行动导向动词：每个要点以强有力的行动动词开头，但根据实习生的实际角色选择合适的动词，如“协助”、“参与”、“支持”、“优化”、“分析”、“协助推动”等，避免过度夸大（如“领导”、“负责整个项目”），除非确有实据。

量化成果：尽可能使用数字、百分比、金额、时间等数据来量化成就（如“协助提升页面点击率15%”、“参与完成XX份市场调研报告，为团队提供决策支持”）。

ATS友好格式：使用简洁的排版，避免表格、图片、复杂格式或特殊字符。使用Markdown语法进行清晰排版（如粗体、列表）。

真实性保障：所有优化内容基于用户提供的原始信息，不虚构经历，但可通过措辞优化突出亮点，强调学习能力、适应能力和对团队的贡献。

篇幅控制：每个经历的描述控制在3-5个要点内，重点突出与JD相关的内容。

技能匹配：根据JD要求重新组织和补充技能，列出与JD高度相关的硬技能和软技能，并可按照熟练度或相关性分组。技能描述需真实反映用户掌握程度。

输出格式：使用二级标题（例如 ## 实习经历、## 项目经验、## 技能清单）分隔三个部分。内容需简洁、专业。

第三部分：面试指导要求
针对优化后的简历，结合公司分析，为用户提供实用的面试准备建议，帮助用户在面试中“自圆其说”，展示最佳状态。指导内容应包含：

简历深挖点提示：指出优化后的简历中哪些经历或成果最可能在面试中被面试官深挖（尤其是实习中参与的具体项目、数据分析、协作细节等），并建议用户如何准备背后的故事（如项目难点、个人贡献、遇到的挑战及如何解决、学到的经验等）。

公司匹配度展示：结合第一部分的公司分析，提示用户在面试中可以主动提及哪些公司关注点，以证明自己与公司和岗位的契合度。例如，可以强调自己对该公司战略方向的理解，并举例说明自己的实习经验或项目如何与公司当前业务相呼应，展示自己的潜力。
常见面试问题及回答思路：列举1-2个与岗位相关的常见面试问题，并提供回答思路或框架，尤其是行为面试问题，建议使用STAR法则回答。

面试实用技巧：作为人力资源专家，补充一些提高面试通过率的通用技巧，特别是针对学生求职者：
如何准备自我介绍（突出与JD最匹配的3个亮点，结合实习经历）。

如何回答“你为什么选择我们公司”这类问题（结合公司分析，展示对公司的研究和认同）。

如何应对“你最大的缺点是什么”等敏感问题（转化为成长性思维）。

如何通过提问展示对岗位的思考（提出有深度的问题，如团队目前面临的挑战、岗位的成长路径等）。
着装、礼仪、心态等软性建议，强调自信和真诚的重要性。

输出格式：使用二级标题 ## 面试指导，内容可分点阐述，语言亲切、实用，符合学生求职场景。

用户输入格式
请提供以下信息，我将为您生成公司分析、优化简历和面试指导：
【职位描述】：（复制粘贴完整的JD，如果知道公司名称，请在描述中体现或注明）
【工作/实习经历】：（请逐段描述，每段包括公司、职位、时间段，以及具体职责和成就。尽量详细，以便提取关键信息）
【项目经验】：（如有，请描述项目名称、角色、时间、职责和成果）
【技能清单】：（列出您的硬技能和软技能，例如：Python、项目管理、团队协作等）

【岗位描述】
{request.job_description}

【原始简历】
{request.resume_text}

【目标公司名称】（如果有）
{request.company_name if request.company_name else "（未提供，请从岗位描述中推断）"}

请严格按照以下格式输出（使用 Markdown 标题分隔）：
## 公司分析
（内容）

## 优化简历
（内容）

## 面试建议
（内容）
"""

    try:
        # 3. 调用 DeepSeek 模型（异步）
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的简历优化助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000,
        )
        tailored_resume = response.choices[0].message.content
        return {"tailored_resume": tailored_resume}

    except openai.AuthenticationError:
        print("!!! API Key 认证失败 !!!", flush=True)
        raise HTTPException(status_code=401, detail="API Key 无效或已过期")
    except openai.RateLimitError:
        print("!!! 速率限制错误 !!!", flush=True)
        raise HTTPException(status_code=429, detail="DeepSeek API 速率限制已达，请稍后重试或检查账户余额")
    except openai.APITimeoutError:
        print("!!! API 超时 !!!", flush=True)
        raise HTTPException(status_code=504, detail="DeepSeek API 超时，请稍后重试")
    except openai.APIError as e:
        print(f"!!! DeepSeek API 错误: {e} !!!", flush=True)
        raise HTTPException(status_code=502, detail=f"DeepSeek API 服务异常: {str(e)}")
    except Exception as e:
        # 生产环境应记录日志
        print(f"!!! 未捕获的异常: {e} !!!", flush=True)
    import traceback
    traceback.print_exc()  # 这会打印完整的错误堆栈
    raise HTTPException(status_code=500, detail="服务器内部错误")

# ==================== 健康检查 ====================
@app.get("/health")
async def health():
    return {"status": "ok"}