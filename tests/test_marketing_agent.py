import requests
import time

API_URL = "http://localhost:8002/bookquery"

def test_bookquery_response():
    payload = {
        "query": "What marketing strategy should I use for the skincare segment?",
        "segment": "Skincare Segment",
        "model": "gemini"
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=30)
    except Exception as e:
        print(f"❌ Could not connect to API server at {API_URL}: {e}")
        assert False, f"API server not reachable: {e}"

    assert response.status_code == 200, f"Non-200 response: {response.status_code} - {response.text}"

    try:
        data = response.json()
    except Exception as e:
        print(f"❌ Response is not valid JSON: {response.text}")
        assert False, "Response is not valid JSON"

    print("Initial API response:", data)

    # Poll for result if status is processing
    session_id = data.get("session_id")
    poll_url = f"http://localhost:8002/bookquery/{session_id}"
    max_retries = 30
    for i in range(max_retries):
        poll_response = requests.get(poll_url)
        poll_data = poll_response.json()
        print(f"Poll attempt {i+1}: {poll_data}")
        if poll_data.get("status") == "completed":
            result_text = poll_data.get("result") or poll_data.get("answer") or poll_data.get("content")
            assert result_text and isinstance(result_text, str) and len(result_text) > 0, "Empty result in response"
            print("✅ API integration test passed. Response received:")
            print(result_text[:500])
            return
        elif poll_data.get("status") == "error":
            assert False, f"API returned error: {poll_data.get('error')}"
        time.sleep(2)
    assert False, "Timed out waiting for completed result"

if __name__ == "__main__":
    test_bookquery_response()