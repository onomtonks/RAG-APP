import time
import json
from openai import OpenAI


def summary(json_chunks):
    """
    Summarize text chunks stored in the JSON file.
    json_chunks: list of dicts like:
    [
        {"cluster": 0, "chunk_text": "..."}
    ]
    """

    # Extract chunk texts
    texts = [item["text"] for item in json_chunks]

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="YOUR_API_KEY"
    )

    def summarize_text(chunk_text):
        prompt = f"""
        You are an assistant tasked with summarizing paragraphs extracted from a PDF.
        Provide a concise, accurate summary.
        Respond ONLY with the summary.

        Chunk content:
        {chunk_text}
        """

        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-nano-8b-v1",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            stream=False
        )

        # FIX: access the content correctly
        return completion.choices[0].message.content.strip()

    summaries = []

    batch_size = 1
    for i in range(0, len(texts), batch_size):
        chunk = texts[i]
        #print(f"\nProcessing chunk {i} (length={len(chunk)})")

        summary_text = summarize_text(chunk)
        summaries.append(summary_text)

        #print("Summary:", summary_text)

        # mild delay for safety
        time.sleep(1)

    # Combine cluster + summary
    final_output = []
    for item, summary_text in zip(json_chunks, summaries):
        final_output.append({
            "cluster": item["cluster_id"],
            "summary": summary_text
        })

    return final_output


if __name__ == "__main__":
    with open("src/final_chunks.json", "r") as f:
        json_chunks = json.load(f)

    summaries = summary(json_chunks)
    print(summaries)
    
