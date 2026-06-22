import os
import json
import asyncio
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.genai import types

from expense_agent.agent import root_agent

def serialize_event(event):
    d = {
        "author": event.author,
    }
    if event.content:
        content_dict = event.content.model_dump(exclude_none=True)
        if "role" not in content_dict:
            content_dict["role"] = getattr(event.content, "role", "model")
        if "parts" in content_dict:
            new_parts = []
            for p in content_dict["parts"]:
                new_p = {}
                if "text" in p:
                    new_p["text"] = p["text"]
                if "function_call" in p or "functionCall" in p:
                    fc = p.get("function_call") or p.get("functionCall")
                    new_p["function_call"] = {
                        "name": fc.get("name"),
                        "args": fc.get("args"),
                        "id": fc.get("id"),
                    }
                if "function_response" in p or "functionResponse" in p:
                    fr = p.get("function_response") or p.get("functionResponse")
                    new_p["function_response"] = {
                        "name": fr.get("name"),
                        "response": fr.get("response"),
                        "id": fr.get("id"),
                    }
                if new_p:
                    new_parts.append(new_p)
            content_dict["parts"] = new_parts
        d["content"] = content_dict
    else:
        if event.output is not None:
            d["content"] = {
                "role": "model",
                "parts": [
                    {
                        "text": f"Node Output: {json.dumps(event.output)}"
                    }
                ]
            }
    return d

async def run_case(case):
    eval_case_id = case["eval_case_id"]
    prompt_text = case["prompt"]["parts"][0]["text"]
    print(f"Running scenario: {eval_case_id}")

    session_service = InMemorySessionService()
    session = session_service.create_session_sync(user_id="test_user", app_name="expense_agent")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="expense_agent")

    msg = types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])
    
    turns = []
    
    # Run 1 (fresh run)
    events_t0 = []
    raw_events_t0 = []
    interrupt_id = None
    async for e in runner.run_async(new_message=msg, user_id="test_user", session_id=session.id, yield_user_message=True):
        raw_events_t0.append(e)
        events_t0.append(serialize_event(e))
        if e.content:
            for p in e.content.parts:
                if p.function_call and p.function_call.name == "adk_request_input":
                    interrupt_id = p.function_call.id
    
    turns.append({
        "turn_index": 0,
        "events": events_t0
    })

    raw_events_t1 = []
    if interrupt_id:
        # Determine human response: reject prompt injection, approve others
        decision = "reject" if eval_case_id == "prompt_injection" else "approve"
        print(f"  Intercepted human-in-the-loop approval. Decision: {decision}")
        
        resume_msg = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=interrupt_id,
                        name="adk_request_input",
                        response={"result": decision}
                    )
                )
            ]
        )
        
        events_t1 = []
        async for e in runner.run_async(new_message=resume_msg, user_id="test_user", session_id=session.id, yield_user_message=True):
            raw_events_t1.append(e)
            events_t1.append(serialize_event(e))
        
        turns.append({
            "turn_index": 1,
            "events": events_t1
        })
    
    # Find the final decision output to populate the candidate responses list
    final_output = None
    for e in reversed(raw_events_t1 or raw_events_t0):
        if e.output and isinstance(e.output, dict) and "decision" in e.output:
            final_output = e.output
            break
            
    response_text = json.dumps(final_output) if final_output else "No outcome recorded"
    
    responses = [
        {
            "response": {
                "role": "model",
                "parts": [
                    {
                        "text": response_text
                    }
                ]
            }
        }
    ]
    
    return {
        "eval_case_id": eval_case_id,
        "prompt": case["prompt"],
        "responses": responses,
        "agent_data": {
            "agents": {
                "ambient_expense_workflow": {
                    "agent_id": "ambient_expense_workflow",
                    "instruction": "Ambient Expense Approval workflow."
                }
            },
            "turns": turns
        }
    }

async def main():
    dataset_path = "tests/eval/datasets/basic-dataset.json"
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    generated_cases = []
    for case in dataset["eval_cases"]:
        generated_case = await run_case(case)
        generated_cases.append(generated_case)
    
    output_dir = "artifacts/traces"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "generated_traces.json")
    with open(output_path, "w") as f:
        json.dump({"eval_cases": generated_cases}, f, indent=2)
    print(f"Saved generated traces to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
