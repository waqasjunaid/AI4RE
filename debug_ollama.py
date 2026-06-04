import json, urllib.request, sys, time

HOST = "http://127.0.0.1:11434"

SYSTEM = (
    "You are a JSON API. Output ONLY a valid JSON array. "
    "No markdown. No explanation. No text before or after the array."
)

USER = (
    "Extract entities from this text. "
    "Return a JSON array where each item has keys: "
    "entity_type, content, confidence_score.\n\n"
    "Valid entity_type values: function_point, constraint, actor_role, exception_condition\n\n"
    "Text: The system shall respond in under 2 seconds. "
    "Administrators can create and delete user accounts. "
    "The system locks accounts after 5 failed login attempts.\n\n"
    "JSON array:"
)


def chat(model, timeout=240):
    url = "{}/api/chat".format(HOST)
    payload = json.dumps({
        "model":  model,
        "stream": True,
        "options": {
            "temperature": 0.0,
            "num_predict": 600,
            "stop": [],
        },
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": USER},
        ],
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    tokens = []
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                tokens.append(d.get("message", {}).get("content", ""))
                if d.get("done"):
                    break
            except Exception:
                continue
    elapsed = time.time() - t0
    return "".join(tokens).strip(), elapsed


def get_models():
    req = urllib.request.Request("{}/api/tags".format(HOST))
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode())
    return [m["name"] for m in data.get("models", [])]


def test_model(model):
    print("=" * 55)
    print("Testing: {}  (timeout=240s)".format(model))
    print("=" * 55)
    try:
        result, elapsed = chat(model, timeout=240)
        print("Response ({:.1f}s):".format(elapsed))
        print(result[:600])
        print()

        bad_signs = [
            "<|im_start|>", "entity_type_type",
            "contentcontent", "entityentity"
        ]
        is_corrupt = any(sign in result for sign in bad_signs)

        # Strip markdown fences
        clean = result
        if "```" in clean:
            lines = clean.splitlines()
            clean = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()

        start = clean.find("[")
        end   = clean.rfind("]")
        is_json_ok = False

        if start != -1 and end != -1 and end > start:
            candidate = clean[start:end+1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list) and len(parsed) > 0:
                    is_json_ok = True
                    print("JSON: OK -- {} items found".format(len(parsed)))
                    for item in parsed[:4]:
                        print("  [{:20s}] {}".format(
                            str(item.get("entity_type","?"))[:20],
                            str(item.get("content","?"))[:55]
                        ))
                else:
                    print("JSON: parsed but empty list")
            except Exception as e:
                print("JSON parse failed:", e)
                print("Candidate:", repr(candidate[:150]))
        else:
            print("No JSON array found in response")

        if is_corrupt:
            print("\nSTATUS: CORRUPTED")
        elif is_json_ok:
            print("\nSTATUS: WORKING -- use this model")
        else:
            print("\nSTATUS: PARTIAL -- model responded but JSON incomplete")
            print("Try: ollama pull {} (get latest version)".format(model))

    except Exception as e:
        print("ERROR:", e)

    print()


print("Available models:")
try:
    models = get_models()
    for m in models:
        print("  ", m)
except Exception as e:
    print("Cannot connect:", e)
    sys.exit(1)

print()

# Test llama3.2:3b first (fastest), then others
order = ["llama3.2:3b", "llama3.1:70b", "qwen2.5:14b"]
tested = set()

for model in order:
    if any(model.split(":")[0] in m for m in models):
        matching = [m for m in models if model.split(":")[0] in m]
        for m in matching:
            if m not in tested:
                test_model(m)
                tested.add(m)