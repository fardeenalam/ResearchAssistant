from tavily import TavilyClient


def web_search(questions: list[str]) -> list[str]:
    tavily_client = TavilyClient(api_key="")

    raw_evidence = []
    for question in questions:
        response = tavily_client.search(query=question)

        contents = []
        for item in response.get("results", []):
            text = item.get("content")
            if text:
                contents.append(text)

        combined = "\n".join(contents)
        raw_evidence.append(combined)

    return raw_evidence