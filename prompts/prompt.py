"""Language-selectable prompts used by the Phase 1 continuation pipeline."""

from __future__ import annotations

import copy
from typing import Dict, List


PHASE1_ASSISTANT_PROMPTS = {
    "en": """
You are an expert literary assistant. Based on the preceding passage, continue the story with the next complete, natural paragraph while preserving the original author's narrative style, tone, syntax, and pacing.

Rules:
1. Continue seamlessly from the preceding passage as if it were written by the same author.
2. Maintain strong narrative logic, plot continuity, and stylistic coherence with the preceding passage.
3. Target approximately 80-130 words for this short paragraph. Prioritize a complete,
   natural ending over an exact length; do not pad or truncate to reach the target.
4. Continue from the endpoint without recapping the preceding passage. Do not
   introduce an unrelated scene or unprepared characters.
5. Output only the next complete paragraph. Do not add conversational replies,
   titles, explanations, or any other meta-commentary.
""",
    "zh": """
你是一位專業文學助手，請根據故事前文，續寫緊接其後的一個小段落，並維持原作者的敘事風格、語氣、句法結構與節奏。

規則：
1. 讓續文自然銜接前文，用字遣詞、劇情、語調、節奏、風格、人物口吻或敘事邏輯上如同由同一位作者寫成。
2. 維持嚴密的敘事邏輯，以及與前文一致的文體。
3. 目標篇幅約為 80-130 字。以敘事完整、自然收束為優先，不要為了達到字數而灌水、重複或硬截斷。
4. 從前文結尾直接往下續寫，不要重述前文，也不要突然引入沒有鋪陳的新場景或人物。
5. 只輸出下一個段落內文；不要加入對話式回覆、標題或任何後設說明。
""",
}


PHASE1_CONTINUATION_USER_PROMPTS = {
    "en": """Below is the preceding passage:

{context}

Continue with the next short, complete, natural paragraph:""",
    "zh": """以下是故事前文：

{context}

請續寫緊接其後的一個小段落：""",
}


PHASE1_USER_PROMPTS = {
    "en": """
You are a professional literary mentor guiding an AI assistant to refine its story continuation. The assistant aims to emulate the original author's style, logic, and narrative rhythm based on the provided "Story Context."
Your task is to compare the "Assistant Draft" against the "Original Completion" within the context of the background and provide concise, actionable feedback.

Rules:
1. Focus on Style & Direction: Highlight specific gaps in diction, tone, pacing, character voice, or logic.
2. Improvement Goal: Guide the assistant toward the target author's style, logic, and narrative rhythm, while making the continuation fit naturally with the story context.
3. Concrete Guidance: Keep feedback highly specific and directly actionable based on the draft; avoid generic criticism.
4. Length Check: The target continuation is approximately 280–350 words. If the draft is clearly below or over this range, explicitly say that it is too short or too long and instruct the assistant to adjust.
5. Do Not Reveal the Answer: Do not directly provide the reference continuation's exact plot events or dialogue; use indirect guidance instead.
6. Keep It Concise: Output only 5–7 direct feedback sentences (under 180 words).
7. Strict Output Format: Output the feedback text only.
""",
    "zh": """
你是一位專業的文學導師，正在引導 AI 助手修改其文章，助手的任務是根據給定的「故事前文」，試圖模擬該作家的文風與邏輯，續寫可能的下一段落，並且內容要高度契合原作者的目標風格與敘事節奏。

而你的任務是幫助這位 AI 助手寫得更好。請在「故事前文」的脈絡下，對比「助手草稿」與「原作續文」。提供簡潔的修訂反饋，以協助助手改進。

規則：
1. 聚焦於風格與方向：指出用字遣詞、語調、節奏、風格、人物口吻或敘事邏輯上的具體差距。
2. 反饋目標：透過此反饋與引導，使助理撰寫的文章更接近原作風格，並且使續寫內容更符合故事前文，彷彿是由同一位作家所撰寫。
3. 反饋必須非常具體、根據其文章內容提供明確修改方向，避免空泛詞彙。
4. 篇幅檢查：續寫目標約為 280-350 字。如果助手草稿明顯太短或太長，應引導其做調整。
5. 避免直接告知原作中的精確情節解答或對話，盡量使用引導方式。
6. 保持簡潔：僅輸出 5-7 句直接的反饋（字數控制在 180 字以內）。
7. 格式限制：僅輸出反饋本文。
""",
}


PHASE1_FEEDBACK_USER_TEMPLATES = {
    "en": """
[Story Context x]
{context}

[Assistant Draft y]
{completion}

[Original Completion y_hat]
{reference}

Write the mentor's next feedback message:
""",
    "zh": """
[故事前文]
{context}

[助手草稿]
{completion}

[原作續文]
{reference}

請寫出文學導師接下來給助手的修改回饋：
""",
}


PHASE1_FEEDBACK_FALLBACKS = {
    "en": "Please revise your continuation based on the specific differences between the target style and your current draft.",
    "zh": "請根據目標原作續文與目前草稿的具體差異，重新修改你的續寫。",
}


PHASE1_JUDGE_SYSTEM_PROMPTS = {
    "en": """
You are a strict but conservative pairwise evaluator of literary continuations.

You will be given the target author, the story context, the author's reference continuation, and two candidates. Determine which candidate is more likely to be a natural continuation in the author's style and narrative voice.

Evaluate the candidates in this order:
1. Context continuity: character state, setting, time, narrative viewpoint, and logical progression from the context.
2. Authorial style: diction, syntax, sentence rhythm, tone, and character voice.
3. Narrative quality: whether the candidate develops the scene naturally rather than adding disconnected events, adjective piling, repetition, or padding.
4. Completeness: penalize only an obviously truncated or unfinished passage, or a passage that does not form a complete paragraph. Do not use a fixed word or token threshold.

Rules:
- Use the reference continuation as a calibration example for the author's style, narrative logic, and likely continuation. It is not an answer key.
- Do not prefer a candidate merely because it copies words, phrases, or plot details from the reference.
- Do not prefer a candidate merely because it is longer. Do not penalize a candidate merely because it is shorter unless it is clearly incomplete.
- If the quality difference is too small or the evidence is insufficient, choose C.
- Complete the comparison across all four criteria, but output only the most
  important reasons. Use at most four sentences for the rationale; do not use
  Markdown, numbered lists, or repeated analysis. Then put the decision on the
  final line, using strictly one of the following formats:
Judgement: A
Judgement: B
Judgement: C

(Judgement: C means the candidates are indistinguishable in quality, or the evidence is insufficient to choose a winner.)
""",
    "zh": """
你是一位嚴格但保守的文學續寫成對評審。

你會收到目標作者、故事前文、原作者的參考續文，以及候選 A 和候選 B。請判斷哪一篇更可能是自然承接前文、並符合原作者文體與敘事聲音的續文。

請依下列順序評估：
1. 前文銜接：人物狀態、場景、時間、敘事視角，以及是否合乎邏輯地承接前文。
2. 作者風格：用字遣詞、句法、句子節奏、語氣與人物口吻。
3. 敘事品質：是否自然發展場景，而不是堆疊形容詞、重複內容、加入無關事件或灌水。
4. 完整度：只有在候選明顯中途截斷、句子未完成或沒有形成完整段落時才扣分，不使用固定字數或 token 門檻。

規則：
- 原作續文只用來校準作者風格、敘事邏輯與可能的承接方式，不是標準答案。
- 不得只因候選複製原作續文的字詞、句子或情節就判定它較好。
- 不得只因候選較長就偏好它；除非明顯未完成，也不要只因候選較短就扣分。
- 如果兩篇品質差異太小，或證據不足以選出勝者，請判定為 C。
- 完成上述四個面向的比較，但只輸出最重要的理由。理由最多四句，不要
  使用 Markdown、編號或重複分析。最後一行必須嚴格輸出下列其中一項：
判定：A
判定：B
判定：C

（判定：C 代表兩篇品質難以區分，或證據不足以選出勝者。）
""",
}


PHASE1_JUDGE_USER_PROMPTS = {
    "en": """
[Target Author]
{author}

[Context]
{context}

[Reference Original Continuation]
{reference}

[Candidate A]
{candidate_a}

[Candidate B]
{candidate_b}

Which candidate is more likely to be a natural continuation in {author}'s style and narrative voice?
Follow the system's evaluation criteria. Give at most four concise rationale sentences with no Markdown or numbered list. Put exactly `Judgement: A`, `Judgement: B`, or `Judgement: C` on the final line.
""",
    "zh": """
[目標作者]
{author}

[故事前文]
{context}

[原作續文]
{reference}

[候選 A]
{candidate_a}

[候選 B]
{candidate_b}

哪一篇候選文更可能是自然承接前文、並符合 {author} 文體與敘事聲音的續文？
請遵守 system prompt 的評分規則。理由最多四句、保持精簡，不要使用 Markdown 或編號。最後一行必須嚴格輸出：判定：A、判定：B 或判定：C。
""",
}


HINDSIGHT_BLOCK_TEMPLATES = {
    "en": (
        "\n\n=== HINDSIGHT CONTEXT ===\n"
        "[The following is a future user message. Use this to guide your answer to the user prompt.]\n"
        "{follow_up}"
    ),
    "zh": (
        "\n\n=== 回溯情境 ===\n"
        "[以下是未來的使用者訊息。請利用這些資訊引導你回應目前的使用者指令。]\n"
        "{follow_up}"
    ),
}


PHASE1_REFERENCE_BLOCK_TEMPLATES = {
    "en": (
        "\n\n[Reference Continuation]\n"
        "{reference}\n\n"
        "You are now the training-time self-teacher. The student saw only the "
        "original story context and cannot see this reference. Use the "
        "reference to reassess the original continuation task: infer the author's "
        "transferable style, narrative state, pacing, and likely constraints. "
        "Then answer the original continuation prompt directly with one natural, "
        "complete paragraph. Do not mention or copy the reference. Do not "
        "output analysis, reasoning, a title, or any meta-commentary:\n"
    ),
    "zh": (
        "\n\n[參考續文]\n"
        "{reference}\n\n"
        "你現在是訓練階段使用的 self-teacher。學生只能看到原本的故事前文，"
        "看不到這份參考續文。請利用參考續文，重新判斷原始續寫任務中可轉移的"
        "作者文體、敘事狀態、節奏與限制，然後直接回答原本的續寫指令，寫出一個自然、"
        "完整的段落。不要提及或複製參考續文，不要輸出分析、思考過程、標題或任何後設說明：\n"
    ),
}


def get_phase1_assistant_prompt(language: str) -> str:
    return PHASE1_ASSISTANT_PROMPTS[language]


def get_phase1_continuation_user_prompt(language: str) -> str:
    return PHASE1_CONTINUATION_USER_PROMPTS[language]


def get_phase1_user_prompt(language: str) -> str:
    return PHASE1_USER_PROMPTS[language]


def get_phase1_feedback_user_template(language: str) -> str:
    return PHASE1_FEEDBACK_USER_TEMPLATES[language]


def get_phase1_feedback_fallback(language: str) -> str:
    return PHASE1_FEEDBACK_FALLBACKS[language]


def get_phase1_judge_system_prompt(language: str) -> str:
    return PHASE1_JUDGE_SYSTEM_PROMPTS[language]


def get_phase1_judge_user_prompt(language: str) -> str:
    return PHASE1_JUDGE_USER_PROMPTS[language]


def get_hindsight_block_template(language: str) -> str:
    return HINDSIGHT_BLOCK_TEMPLATES[language]


def get_phase1_reference_block_template(language: str) -> str:
    return PHASE1_REFERENCE_BLOCK_TEMPLATES[language]


def build_hindsight_messages(
    messages: List[Dict[str, str]],
    follow_up: str,
    language: str = "en",
) -> List[Dict[str, str]]:
    """Append a language-specific hindsight block to the last user message."""
    conditional = copy.deepcopy(messages)
    block = get_hindsight_block_template(language).format(
        follow_up=follow_up.strip(),
    )
    for i in range(len(conditional) - 1, -1, -1):
        if conditional[i]["role"] == "user":
            conditional[i]["content"] += block
            break
    return conditional


def build_phase1_teacher_messages(
    messages: List[Dict[str, str]],
    reference: str,
    language: str = "en",
) -> List[Dict[str, str]]:
    """Add a reference continuation to the prompt used for teacher scoring."""
    conditional = copy.deepcopy(messages)
    block = get_phase1_reference_block_template(language).format(
        reference=reference.strip(),
    )
    for i in range(len(conditional) - 1, -1, -1):
        if conditional[i]["role"] == "user":
            conditional[i]["content"] += block
            break
    return conditional

def build_phase1_messages(
    p_context: str,
    language: str = "en",
) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": get_phase1_assistant_prompt(language).strip(),
        },
        {
            "role": "user",
            "content": get_phase1_continuation_user_prompt(language).format(
                context=p_context.strip()
            ),
        },
    ]

def build_phase1_judge_messages(
    context: str,
    reference: str,
    candidate_a: str,
    candidate_b: str,
    author: str = "Unknown author",
    language: str = "en",
) -> List[Dict[str, str]]:
    """Build the shared Phase 1 reference-based judge messages."""
    user_msg = get_phase1_judge_user_prompt(language).format(
        author=author,
        context=context,
        reference=reference,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
    ).strip()
    return [
        {"role": "system", "content": get_phase1_judge_system_prompt(language).strip()},
        {"role": "user", "content": user_msg},
    ]
