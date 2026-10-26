"""Language-selectable prompts used by the Phase 1 continuation pipeline."""

from __future__ import annotations

import copy
from typing import Dict, List


PHASE1_ASSISTANT_PROMPTS = {
    "en": """
You are an expert literary assistant. Based on the preceding passage, continue the story with the next complete, natural paragraph while preserving the original author's narrative style, tone, syntax, and pacing.

Rules:
1. Continue seamlessly from the preceding passage as if it were written by the same author.
2. Maintain strong narrative logic and stylistic coherence with the preceding passage.
3. Keep the continuation to approximately 250 tokens.
4. Output exactly one paragraph. Do not use blank lines or additional paragraph breaks.
5. Output only the next complete paragraph. Do not add conversational replies, titles, explanations, or any other meta-commentary.
""",
    "zh": """
你是一位專業文學助手，請根據故事前文，續寫緊接其後的一個完整自然段落，並維持原作者的敘事風格、語氣、句法結構與節奏。

規則：
1. 讓續文自然銜接前文，如同由同一位作者寫成。
2. 維持嚴密的敘事邏輯，以及與前文一致的文體。
3. 篇幅控制在約 250 tokens 左右。
4. 全文只能有一個段落，不要使用空白行或額外換行。
5. 只輸出下一個完整段落；不要加入對話式回覆、標題或任何後設說明。
""",
}


PHASE1_CONTINUATION_USER_PROMPTS = {
    "en": """Below is the preceding passage:

{context}

Continue with the next complete, natural paragraph:""",
    "zh": """以下是故事前文：

{context}

請續寫緊接其後的一個完整自然段落：""",
}


PHASE1_USER_PROMPTS = {
    "en": """
You are a professional literary mentor guiding an AI assistant to refine its story continuation. The assistant aims to emulate the original author's style, logic, and narrative rhythm based on the provided "Story Context."
Your task is to compare the "Assistant Draft" against the "Original Completion" within the context of the background and provide concise, actionable feedback.

Rules:
1. Focus on Style & Direction: Highlight specific gaps in diction, tone, pacing, character voice, or logic.
2. Concrete Guidance: Keep feedback highly specific and directly actionable based on the draft; avoid generic criticism.
3. Keep It Concise: Output only 3–5 direct feedback sentences (under 150 words).
4. Strict Output Format: Output the feedback text only.
""",
    "zh": """
你是一位專業的文學導師，正在引導 AI 助手修改其文章，助手的任務是根據給定的「故事前文」，試圖模擬該作家的文風與邏輯，續寫可能的下一段落，並且內容要高度契合原作者的目標風格與敘事節奏。

而你的任務是幫助這位 AI 助手寫得更好。請在「故事前文」的脈絡下，對比「助手草稿」與「原作續文」。提供簡潔的修訂反饋，以協助助手改進。

規則：
1. 聚焦於風格與方向：指出用字遣詞、語調、節奏、風格、人物口吻或敘事邏輯上的具體差距。
2. 反饋目標：透過此反饋與引導，使助理撰寫的文章更接近原作風格，並且使續寫內容更符合故事前文，彷彿是由同一位作家所撰寫。
3. 反饋必須非常具體、根據其文章內容提供明確修改方向，避免空泛詞彙。
4. 嚴禁直接告知原作中的精確情節解答或對話，盡量使用引導方式。
5. 保持簡潔：僅輸出 3-5 句直接的反饋（字數控制在 150 字以內）。
6. 格式限制：僅輸出反饋本文。
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
You are a strict pairwise evaluator of literary continuations for style alignment.

You will be given the name of the target author, the story context, and the reference continuation written by the author.
Your task is to compare Candidate A and Candidate B and determine which candidate is more likely to have been written by the specified target author in terms of both style and narrative trajectory.

Evaluate the candidates based on:
1. Narrative & Plot Continuity: Seamless flow and logical plot progression from the context.
2. Authorial Style & Tone: Alignment with the target author's distinct tone, vocabulary, syntax, and sentence structure.
3. Imagery & Atmosphere: Consistency with the author's descriptive habits, sensory details, and thematic mood.
4. Pacing & Structure: Sentence-level rhythm and paragraph organization typical of the author.

Rules:
- Do NOT simply prefer a candidate because it copies exact words from the reference.
- Prioritize whether the diction, syntax, narrative pacing, character voice, and plot transitions match the original author's style, rather than the word count or paragraph count.
- Reason briefly, then end with strictly one of the following decisions:
Judgement: A
Judgement: B
Judgement: C

(Judgement: C means both candidates are indistinguishable in style quality, or evidence is insufficient to choose a winner.)
""",
    "zh": """
你是一位嚴格的文學續寫成對評審，負責判斷候選續文與目標作者風格的一致性。

你會收到目標作者姓名、故事前文，以及由該作者寫成的原作續文。請比較候選 A 與候選 B，判斷哪一篇在文體與敘事走向上更可能出自指定作者之手。

請依據下列面向評估：
1. 敘事與情節銜接：是否能自然承接前文，並合乎邏輯地推進情節。
2. 作者風格與語氣：是否符合該作者獨特的語氣、詞彙、句法與句子結構。
3. 意象與氛圍：是否符合該作者慣用的描寫方式、感官細節與主題氛圍。
4. 節奏與結構：是否具備該作者典型的句子節奏與段落組織。

規則：
- 不得因候選文較長、細節較多或語言較華麗而偏好該候選，應以哪一篇更準確捕捉原作者真正的寫作風格為判準。
- 優先判斷措辭、句法、敘事節奏、人物語氣與情節銜接是否符合原作者風格，而不是文字數量、段落數量。
- 簡短說明理由後，最後一行必須嚴格輸出下列其中一項：
判定：A
判定：B
判定：C

（判定：C 代表兩篇在風格品質上難以區分，或證據不足以選出勝者。）
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

Which candidate is more likely to be the continuation written by {author}?
Provide a brief rationale analyzing their stylistic alignment, then end with exactly: Judgement: A, Judgement: B, or Judgement: C.
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

哪一篇候選文更可能是由 {author} 寫成的下一段續文？
請簡短說明兩者與作者風格的契合程度，最後必須嚴格輸出：判定：A、判定：B 或判定：C。
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
