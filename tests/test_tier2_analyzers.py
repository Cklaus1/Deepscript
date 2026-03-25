"""Tests for Tier 2 analyzers — pitch, recruiting, management, operations, customer, education."""

import json
from pathlib import Path

from deepscript.analyzers.pitch import PitchAnalyzer
from deepscript.analyzers.recruiting import RecruiterScreenAnalyzer
from deepscript.analyzers.management import ManagementAnalyzer
from deepscript.analyzers.operations import OperationsAnalyzer
from deepscript.analyzers.customer import CustomerAnalyzer
from deepscript.analyzers.education import EducationAnalyzer

FIXTURES = Path(__file__).parent / "fixtures"


def _make_transcript(text, segments=None):
    return {"text": text, "segments": segments or []}


def test_pitch_analyzer():
    transcript = _make_transcript(
        "Thank you for taking the pitch. Our traction shows 200% growth. "
        "Tell me more about your unit economics. What's your burn rate? "
        "Can you send me the data room? I'd like to introduce you to our partner.",
    )
    analyzer = PitchAnalyzer(llm=None)
    result = analyzer.analyze(transcript)
    assert result.call_type == "investor-pitch"
    assert len(result.sections.get("interest_signals", [])) > 0


def test_pitch_detects_concerns():
    transcript = _make_transcript(
        "I'm worried about the competition in this space. "
        "What's your defensibility? The market size seems small.",
    )
    analyzer = PitchAnalyzer(llm=None)
    result = analyzer.analyze(transcript)
    assert len(result.sections.get("investor_concerns", [])) > 0


def test_recruiter_screen():
    transcript = _make_transcript(
        "Tell me about your background in software engineering. "
        "I have 5 years of experience and proficiency in Python. "
        "What are your salary expectations? Looking for 150k base.",
    )
    analyzer = RecruiterScreenAnalyzer(llm=None)
    result = analyzer.analyze(transcript)
    assert result.call_type == "recruiter-screen"
    assert result.sections.get("compensation_discussed") is True


def test_management_1on1():
    transcript = _make_transcript(
        "How are you doing? I've been feeling overwhelmed with the workload. "
        "Let's talk about your career growth and development goals. "
        "I'm blocked on the API integration, waiting on the backend team.",
    )
    analyzer = ManagementAnalyzer(llm=None, meeting_type="one-on-one")
    result = analyzer.analyze(transcript)
    assert result.call_type == "one-on-one"
    assert result.sections["topic_coverage"]["blockers"] > 0
    assert len(result.sections.get("burnout_indicators", [])) > 0


def test_management_cofounder():
    transcript = _make_transcript(
        "I think we're misaligned on our strategy. The vision needs to be clearer. "
        "I disagree with the direction we're taking on the product.",
    )
    analyzer = ManagementAnalyzer(llm=None, meeting_type="cofounder-alignment")
    result = analyzer.analyze(transcript)
    assert result.call_type == "cofounder-alignment"
    assert len(result.sections.get("alignment_signals", [])) > 0


def test_operations_standup():
    segments = [
        {"start": 0, "end": 30, "text": "Yesterday I worked on the login page.", "speaker": "Dev1"},
        {"start": 30, "end": 60, "text": "I'm blocked on the API, waiting on backend.", "speaker": "Dev2"},
    ]
    transcript = _make_transcript("", segments)
    transcript["text"] = " ".join(s["text"] for s in segments)
    analyzer = OperationsAnalyzer(llm=None, ops_type="standup")
    result = analyzer.analyze(transcript)
    assert result.call_type == "standup"
    assert len(result.sections.get("blockers_detected", [])) > 0


def test_operations_retro():
    transcript = _make_transcript(
        "What went well this sprint? The deploy process was great. "
        "What should we improve? The code review process is frustrating.",
    )
    analyzer = OperationsAnalyzer(llm=None, ops_type="sprint-retro")
    result = analyzer.analyze(transcript)
    assert result.call_type == "sprint-retro"
    assert len(result.sections.get("went_well", [])) > 0
    assert len(result.sections.get("to_improve", [])) > 0


def test_operations_postmortem():
    transcript = _make_transcript(
        "The root cause was a misconfigured load balancer. "
        "This was caused by a deployment script error. "
        "The outage lasted 2 hours.",
    )
    analyzer = OperationsAnalyzer(llm=None, ops_type="postmortem")
    result = analyzer.analyze(transcript)
    assert result.call_type == "postmortem"
    assert len(result.sections.get("root_cause_signals", [])) > 0
    assert result.sections.get("blamelessness") == "healthy"


def test_customer_onboarding():
    transcript = _make_transcript(
        "Welcome to the platform! Let's get you set up. "
        "I'm confused about how to configure the dashboard. "
        "Where is the settings page? I can't find it.",
    )
    analyzer = CustomerAnalyzer(llm=None, cs_type="customer-onboarding")
    result = analyzer.analyze(transcript)
    assert result.call_type == "customer-onboarding"
    assert len(result.sections.get("confusion_points", [])) > 0


def test_customer_churn():
    transcript = _make_transcript(
        "We've decided to cancel our subscription. "
        "The product is too expensive and doesn't meet our needs. "
        "We're switching to a competitor.",
    )
    analyzer = CustomerAnalyzer(llm=None, cs_type="churn-save")
    result = analyzer.analyze(transcript)
    assert result.call_type == "churn-save"
    assert len(result.sections.get("churn_signals", [])) > 0


def test_education_classroom():
    transcript = _make_transcript(
        "Today we'll cover binary search algorithms. "
        "Does everyone understand the concept? I'm confused about the base case. "
        "Let me repeat: the base case is when the array is empty.",
    )
    analyzer = EducationAnalyzer(llm=None, edu_type="classroom")
    result = analyzer.analyze(transcript)
    assert result.call_type == "classroom"
    assert result.sections.get("comprehension_signals", 0) > 0


def test_education_coaching():
    transcript = _make_transcript(
        "What's your goal for this quarter? I want to achieve a promotion. "
        "What's your current situation? Right now I'm a senior engineer. "
        "What options do you have? I could lead a project or mentor juniors. "
        "What will you commit to doing? I will do the tech lead rotation by next month.",
    )
    analyzer = EducationAnalyzer(llm=None, edu_type="coaching-session")
    result = analyzer.analyze(transcript)
    assert result.call_type == "coaching-session"
    grow = result.sections.get("grow_model", {})
    assert grow.get("goal", 0) > 0
    assert grow.get("will", 0) > 0
