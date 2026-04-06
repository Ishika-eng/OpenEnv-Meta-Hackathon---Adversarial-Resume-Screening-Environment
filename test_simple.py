import requests
import json

BASE_URL = "https://ishikamahadar-resume-env.hf.space"


def test_environment():
    print(f"--- Testing Environment at: {BASE_URL} ---")

    # 1. Health check
    print("\n[1/4] Health check...")
    health_resp = requests.get(f"{BASE_URL}/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.status_code}"
    print("Health check passed.")

    # 2. Reset
    print("\n[2/4] Resetting environment...")
    reset_resp = requests.post(f"{BASE_URL}/reset")
    assert reset_resp.status_code == 200, f"Reset failed: {reset_resp.status_code}"
    obs = reset_resp.json()
    obs_data = obs.get("observation", obs)
    print(f"Task type: {obs_data.get('task_type')}")
    print(f"Steps remaining: {obs_data.get('steps_remaining')}")
    print(f"Available actions: {obs_data.get('available_actions')}")

    # 3. Investigation step: view experience
    print("\n[3/4] Viewing experience section...")
    step_resp = requests.post(f"{BASE_URL}/step", json={
        "action": {
            "action_type": "view_section",
            "section": "experience"
        }
    })
    assert step_resp.status_code == 200, f"Step failed: {step_resp.status_code}"
    step_data = step_resp.json()
    step_obs = step_data.get("observation", step_data)
    print(f"Reward: {step_obs.get('reward')}")
    print(f"Sections visible: {list(step_obs.get('visible_sections', {}).keys())}")

    # 4. Submit decision
    print("\n[4/4] Submitting decision...")
    decision_resp = requests.post(f"{BASE_URL}/step", json={
        "action": {
            "action_type": "submit_decision",
            "decision": "accept",
            "fraud_flag": False,
            "confidence": 0.8,
            "fraud_reasoning": ""
        }
    })
    assert decision_resp.status_code == 200, f"Decision step failed: {decision_resp.status_code}"
    dec_data = decision_resp.json()
    dec_obs = dec_data.get("observation", dec_data)
    print(f"Done: {dec_obs.get('done')}")
    print(f"Final reward: {dec_obs.get('reward')}")
    print(f"Phase: {dec_obs.get('phase')}")

    print("\n--- All tests passed! ---")


if __name__ == "__main__":
    test_environment()
