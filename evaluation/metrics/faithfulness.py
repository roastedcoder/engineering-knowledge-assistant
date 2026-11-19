import json


class FaithfulnessEvaluator:

    def __init__(self, llm_service):
        self.llm_service = llm_service

    def evaluate(self, context: str, answer: str) -> float:
        if not context.strip():
            raise ValueError("Context cannot be empty")

        if not answer.strip():
            raise ValueError("Answer cannot be empty")

        system_prompt = """
        You are an answer faithfulness evaluator.

        Your task is to evaluate whether each factual claim
        in the answer is supported by the provided context.

        Rules:
        - Use only the provided context.
        - Do not use external knowledge.
        - Break the answer into individual factual claims.
        - Mark a claim supported only if the context supports it.
        - Return valid JSON only, without markdown.

        Required JSON format:
        {
            "claims": [
                {
                    "claim": "A factual claim",
                    "supported": true
                }
            ]
        }
        """

        prompt = f"""
        Context:
        {context}

        Answer:
        {answer}

        Evaluate the answer's factual claims.
        """

        response = self.llm_service.generate(
            prompt=prompt,
            system_prompt=system_prompt,
        )

        result = json.loads(response)

        claims = result.get("claims")

        if not isinstance(claims, list) or not claims:
            raise ValueError("Evaluator must return at least one claim")

        for claim in claims:
            if (
                not isinstance(claim, dict)
                or not isinstance(claim.get("claim"), str)
                or not claim["claim"].strip()
                or type(claim.get("supported")) is not bool
            ):
                raise ValueError("Invalid claim format returned by evaluator")

        supported_count = sum(
            claim["supported"] for claim in claims
        )

        return supported_count / len(claims)