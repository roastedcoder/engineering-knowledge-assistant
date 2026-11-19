import json
import re


class CitationEvaluator:

    def __init__(self, llm_service):
        self.llm_service = llm_service

    def evaluate(self, context: str, answer: str) -> float:
        if not context.strip():
            raise ValueError("Context cannot be empty")

        if not answer.strip():
            raise ValueError("Answer cannot be empty")

        available_ids = {
            int(source_id)
            for source_id in re.findall(r"(?m)^\[(\d+)\]", context)
        }

        system_prompt = """
        You are a citation accuracy evaluator.

        Evaluate whether the citations attached to each factual
        claim in the answer actually support that claim.

        Rules:
        - Use only the provided context.
        - Identify every factual claim in the answer.
        - Extract the citation IDs attached to each claim.
        - Mark supported as true only if at least one cited
          source directly supports the claim.
        - Do not use external knowledge.
        - Return valid JSON only, without markdown.

        Required JSON format:
        {
            "claims": [
                {
                    "claim": "A factual claim",
                    "citations": [1],
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

        Evaluate the citations in the answer.
        """

        response = self.llm_service.generate(
            prompt=prompt,
            system_prompt=system_prompt,
        )

        result = json.loads(response)
        claims = result.get("claims")

        if not isinstance(claims, list) or not claims:
            raise ValueError("Evaluator must return at least one claim")

        valid_claims = 0

        for claim in claims:
            if (
                not isinstance(claim, dict)
                or not isinstance(claim.get("claim"), str)
                or not claim["claim"].strip()
                or not isinstance(claim.get("citations"), list)
                or type(claim.get("supported")) is not bool
            ):
                raise ValueError("Invalid claim format returned by evaluator")

            citations = claim["citations"]

            if not all(
                type(citation_id) is int
                for citation_id in citations
            ):
                raise ValueError("Citation IDs must be integers")

            has_valid_source = any(
                citation_id in available_ids
                for citation_id in citations
            )

            if claim["supported"] and has_valid_source:
                valid_claims += 1

        return valid_claims / len(claims)