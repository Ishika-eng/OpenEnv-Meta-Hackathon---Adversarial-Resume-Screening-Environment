"""
Inference Script for Resume Screening Environment
===================================
MANDATORY
- Environment variables required:
    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face / API key (or API_KEY)
    
- The inference script must be named `inference.py`
- Must use OpenAI Client for all LLM calls

STDOUT FORMAT
- Exactly three line types to stdout:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin
    - One [STEP] line per step, immediately after env.step()
    - One [END] line after env.close(), always emitted
    - reward and rewards formatted to 2 decimal places
    - done and success are lowercase booleans: true or false
    - error is raw error string or null
    - All fields on single line
    - Score in [0, 1]
"""

import asyncio
import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI
from dotenv import load_dotenv

from client import ResumeEnv
from models import ResumeAction

# Load environment variables
load_dotenv()

# Environment Configuration
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
# API_BASE_URL is for the LLM provider (e.g. OpenAI or HF Router)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
# ENV_URL is for your specific OpenEnv environment server
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

# Task Configuration
TASK_NAME = os.getenv("RESUME_TASK", "resume-screening")
BENCHMARK = os.getenv("RESUME_BENCHMARK", "resume-eval")
NUM_EPISODES = 5  # Changed from hardcoded to configurable

# Model Parameters
TEMPERATURE = 0.7
MAX_TOKENS = 500

# Scoring Configuration
SUCCESS_SCORE_THRESHOLD = 0.6  # 60% accuracy needed for success
MAX_REWARD_PER_EPISODE = 1.0  # Assuming each correct decision = 1.0 reward
MAX_TOTAL_REWARD = NUM_EPISODES * MAX_REWARD_PER_EPISODE

# System Prompt
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert recruitment specialist. Your task is to evaluate resumes against job descriptions.
    
    For each resume, you must:
    1. Determine if the candidate should be ACCEPTED or REJECTED
    2. Detect any potential FRAUD indicators
    3. Provide a CONFIDENCE score for your decision
    
    Guidelines:
    - Accept candidates who closely match the job requirements
    - Reject candidates with significant skill gaps or mismatches
    - Flag fraud if you detect fabricated experience, fake credentials, or inconsistencies
    - Confidence should reflect how certain you are (0.0 = very uncertain, 1.0 = absolutely certain)
    
    Respond ONLY with valid JSON in this exact format:
    {
        "decision": "accept" or "reject",
        "fraud_flag": true or false,
        "confidence": 0.0 to 1.0
    }
    """
).strip()


# ============================================================================
# LOGGING FUNCTIONS (Mandatory Format)
# ============================================================================

def log_start(task: str, env: str, model: str) -> None:
    """Log episode start in required format"""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Log each step in required format"""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log episode end in required format"""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(
        f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ============================================================================
# PROMPT BUILDING
# ============================================================================

def build_user_prompt(
    episode: int,
    job_description: str,
    resume_text: str,
    history: List[str]
) -> str:
    """Build user prompt with job description and resume"""
    history_block = "\n".join(history[-3:]) if history else "None"
    
    return textwrap.dedent(
        f"""
        Episode: {episode}
        
        Job Description:
        {job_description}
        
        Resume Text:
        {resume_text}
        
        Previous Evaluations:
        {history_block}
        
        Evaluate this resume and provide your decision in JSON format.
        """
    ).strip()


# ============================================================================
# LLM INTERACTION
# ============================================================================

def get_model_decision(
    client: OpenAI,
    episode: int,
    job_description: str,
    resume_text: str,
    history: List[str]
) -> dict:
    """
    Get structured decision from LLM
    
    Returns:
        dict with keys: decision, fraud_flag, confidence
        Falls back to safe default on any error
    """
    # Fallback action (safe default - reject with low confidence)
    fallback_action = {
        "decision": "reject",
        "fraud_flag": False,
        "confidence": 0.5
    }
    
    user_prompt = build_user_prompt(episode, job_description, resume_text, history)
    
    try:
        # Check API key validity
        if not API_KEY or API_KEY == "your_api_key":
            print(f"[DEBUG] Invalid API key, using fallback", flush=True)
            return fallback_action
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},  # Force JSON output
            stream=False,
        )
        
        # Parse response
        content_str = response.choices[0].message.content or "{}"
        content = json.loads(content_str)
        
        # Validate response structure
        decision = content.get("decision", "").lower()
        confidence = content.get("confidence", 0.5)
        fraud_flag = content.get("fraud_flag", False)
        
        # Validation checks
        if decision not in ["accept", "reject"]:
            print(f"[DEBUG] Invalid decision: {decision}, using fallback", flush=True)
            return fallback_action
        
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            print(f"[DEBUG] Invalid confidence: {confidence}, using fallback", flush=True)
            return fallback_action
        
        if not isinstance(fraud_flag, bool):
            print(f"[DEBUG] Invalid fraud_flag: {fraud_flag}, using fallback", flush=True)
            return fallback_action
        
        # Valid response
        print(f"[DEBUG] Valid LLM response: {decision}, confidence={confidence:.2f}", flush=True)
        return {
            "decision": decision,
            "fraud_flag": fraud_flag,
            "confidence": float(confidence)
        }
        
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e}, using fallback", flush=True)
        return fallback_action
    except Exception as e:
        print(f"[DEBUG] LLM request failed: {e}, using fallback", flush=True)
        return fallback_action


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main() -> None:
    """Main execution loop following benchmark format"""
    
    # Initialize OpenAI client (Points to LLM)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Initialize environment (Points to OpenEnv Server)
    env = ResumeEnv(base_url=ENV_URL)
    
    # Tracking variables
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    # Log episode start
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        # ================================================================
        # MAIN EPISODE LOOP (5 steps = 5 resume evaluations)
        # ================================================================
        for episode in range(1, NUM_EPISODES + 1):
            
            # Reset environment to get new resume
            observation = await env.reset()
            
            # Extract observation data
            job_description = observation.job_description
            resume_text = observation.resume_text
            
            # Get LLM decision
            action_dict = get_model_decision(
                client=client,
                episode=episode,
                job_description=job_description,
                resume_text=resume_text,
                history=history
            )
            
            # Create action object
            action = ResumeAction(
                decision=action_dict["decision"],
                fraud_flag=action_dict["fraud_flag"],
                confidence=action_dict["confidence"]
            )
            
            # Format action string for logging
            action_str = f"{action.decision}(fraud={action.fraud_flag},conf={action.confidence:.2f})"
            
            # Take step in environment
            observation = await env.step(action)
            
            # Extract results
            reward = observation.reward if hasattr(observation, 'reward') else 0.0
            done = observation.done if hasattr(observation, 'done') else (episode == NUM_EPISODES)
            error = observation.error if hasattr(observation, 'error') else None
            
            # Track metrics
            rewards.append(reward)
            steps_taken = episode
            
            # Log step
            log_step(
                step=episode,
                action=action_str,
                reward=reward,
                done=done,
                error=error
            )
            
            # Update history
            history.append(
                f"Episode {episode}: {action.decision} (conf={action.confidence:.2f}) -> reward={reward:+.2f}"
            )
            
            # Check if episode should end early
            if done:
                break
        
        # ================================================================
        # CALCULATE FINAL SCORE
        # ================================================================
        # Normalize total reward to [0, 1]
        total_reward = sum(rewards)
        score = total_reward / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
        
        # Determine success
        success = score >= SUCCESS_SCORE_THRESHOLD
        
    except Exception as e:
        print(f"[DEBUG] Critical error in main loop: {e}", flush=True)
        success = False
    
    finally:
        # ================================================================
        # CLEANUP & FINAL LOGGING
        # ================================================================
        try:
            # Close environment if it has cleanup method
            if hasattr(env, 'close'):
                await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        
        # Always log end (even on exception)
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    asyncio.run(main())
