# Investigating the effect of truncation on the accuracy of finetuning LLM

## 1. Truncating from Left or Right
Is truncating from L or R better generally?
- llm
- llm finetuned with n/2 from left truncated
- llm finetuned with n/2 from right truncated

**Innovative Approach**: This is a clever way to assess how the position of truncation impacts model understanding and performance. Given that language models might rely more heavily on either the beginning or the end of the input for context (depending on the task and the model architecture), your investigation could reveal important insights about how these models process information.

**Domain Relevance**: The effectiveness of left vs. right truncation might vary across different domains or types of text (e.g., news articles vs. conversational text). It would be interesting to see if domain-specific patterns emerge.

## 2. Linearity of Truncation Impact
Does truncating worsen accuracy linearly?
- llm
- llm finetuned with 1 trunctated
- llm finetuned with 2 trunctated
- llm finetuned with 3 trunctated

**Quantitative Analysis**: This aims to quantify the degradation of model performance as a function of the amount of data truncated, providing a clear, measurable insight into how data reduction affects model outputs. It's a strong approach that could yield highly actionable insights for those working with constrained input sizes.

**Consideration for Non-linearity**: It's possible the relationship isn't linear, especially as some information might be more critical to model performance than others. It would be insightful to explore and model the exact nature of this relationship.

## 3. Comparative Analysis of Truncation Direction
Is L or R better? does it meet hypothesis from 1.?
- llm
- llm finetuned with 1 trunctated from L
- llm finetuned with 1 trunctated from R
- llm finetuned with 2 trunctated from L
- llm finetuned with 2 trunctated from R
- llm finetuned with 3 trunctated from L
- llm finetuned with 3 trunctated from R

**Hypothesis Testing**: This part seeks to validate initial hypotheses and provides a direct comparison that could be extremely valuable for fine-tuning strategies. It's well-structured for drawing clear conclusions about the best practices for data truncation.

**Practical Implications**: The results could have immediate practical implications, especially for applications where input length is a limiting factor, such as tweet generation, SMS-based systems, or other short-form content generation tasks.