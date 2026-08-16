"""Tests for Ephemeral Deep Video Q&A (youtube_ask_video) and VideoAnswerResponse models."""

from src.models import CreatorAttribution, VideoAnswerResponse, VideoEvidence
from src.transcribe import TranscriptionEngine


class TestVideoAnswerModel:
    """Verify VideoAnswerResponse and VideoEvidence models."""

    def test_answer_response_structure(self):
        """VideoAnswerResponse must enforce query, answer, evidence, and attribution."""
        evidence = [
            VideoEvidence(
                timestamp_seconds=920,
                timestamp_formatted="00:15:20",
                speaker_name="Martin Fowler",
                quote="Monoliths become problematic when teams cannot deploy independently.",
                youtube_link="https://www.youtube.com/watch?v=test_vid_001&t=920",
            )
        ]
        attribution = CreatorAttribution(
            channel_title="Tech Talks",
            channel_id="UCtech123456",
            channel_url="https://www.youtube.com/channel/UCtech123456",
            video_url="https://www.youtube.com/watch?v=test_vid_001",
        )
        response = VideoAnswerResponse(
            video_id="test_vid_001",
            title="Microservices Architecture",
            query="Why did the speaker criticize monolithic architectures?",
            answer="The speaker explained that monolithic architectures impede independent team deployments and scale bottlenecks.",
            evidence=evidence,
            attribution=attribution,
        )

        assert response.video_id == "test_vid_001"
        assert response.query == "Why did the speaker criticize monolithic architectures?"
        assert len(response.evidence) == 1
        assert response.evidence[0].timestamp_seconds == 920
        assert response.evidence[0].youtube_link == "https://www.youtube.com/watch?v=test_vid_001&t=920"
        assert response.attribution.channel_title == "Tech Talks"

    def test_answer_serialization_roundtrip(self):
        """VideoAnswerResponse survives JSON roundtrip."""
        evidence = [
            VideoEvidence(
                timestamp_seconds=60,
                timestamp_formatted="00:01:00",
                speaker_name="Host",
                quote="Key takeaway here.",
                youtube_link="https://www.youtube.com/watch?v=test_vid_002&t=60",
            )
        ]
        attribution = CreatorAttribution(
            channel_title="Creator",
            channel_id="UCabc123456",
            channel_url="https://www.youtube.com/channel/UCabc123456",
            video_url="https://www.youtube.com/watch?v=test_vid_002",
        )
        resp = VideoAnswerResponse(
            video_id="test_vid_002",
            query="Test query?",
            answer="Test answer.",
            evidence=evidence,
            attribution=attribution,
            tokens_input=100,
            tokens_output=50,
            cost_usd=0.0001,
        )
        data = resp.model_dump()
        restored = VideoAnswerResponse.model_validate(data)
        assert restored.video_id == resp.video_id
        assert restored.query == resp.query
        assert restored.answer == resp.answer
        assert len(restored.evidence) == 1
        assert restored.tokens_input == 100


class TestAskVideoParser:
    """Verify prompt building and response parsing for ask_video."""

    def setup_method(self):
        self.engine = TranscriptionEngine()

    def test_build_ask_video_prompt_contains_query(self):
        prompt = self.engine._build_ask_video_prompt("What was the outcome of the experiment?")
        assert "<user_query>\nWhat was the outcome of the experiment?\n</user_query>" in prompt
        assert "DIRECT ANSWER" in prompt
        assert "KEY EVIDENCE" in prompt
        assert "Fair Use" in prompt

    def test_parse_ask_video_response_with_smart_quotes(self):
        from src.models import VideoDetailedMetadata
        meta = VideoDetailedMetadata(
            video_id="test1234567",
            title="Quantum Computing 101",
            published_at="2025-01-01T00:00:00Z",
            duration_seconds=1800,
            duration_formatted="00:30:00",
            channel_id="UCscience123",
            channel_title="Science Hub",
        )
        llm_output = (
            "## DIRECT ANSWER\n"
            "The speaker concluded that error correction is the primary bottleneck for quantum scalability.\n\n"
            "## KEY EVIDENCE\n"
            '[00:05:30] Dr. Alice: “Qubit coherence times have improved, but fault-tolerant logical qubits remain the challenge.”\n'
            "[00:12:15] Dr. Bob: 'We expect practical advantage within the decade.'\n"
        )
        parsed = self.engine._parse_ask_video_response(llm_output, "What is the quantum bottleneck?", meta)
        assert parsed.video_id == "test1234567"
        assert "error correction" in parsed.answer
        assert len(parsed.evidence) == 2
        assert parsed.evidence[0].timestamp_seconds == 330
        assert parsed.evidence[0].speaker_name == "Dr. Alice"
        assert "Qubit coherence times" in parsed.evidence[0].quote
        assert parsed.evidence[1].timestamp_seconds == 735
        assert parsed.evidence[1].speaker_name == "Dr. Bob"

    def test_parse_ask_video_response_without_direct_answer_header(self):
        from src.models import VideoDetailedMetadata
        meta = VideoDetailedMetadata(
            video_id="test8888888",
            title="System Architecture",
            published_at="2025-01-01T00:00:00Z",
            duration_seconds=3600,
            duration_formatted="01:00:00",
            channel_id="UCarch123",
            channel_title="Architecture Channel",
        )
        llm_output = (
            "Event-driven architecture provides strong decoupling between producer and consumer services.\n\n"
            "### KEY EVIDENCE\n"
            '[00:10:00] Lead Architect: "Events are immutable logs of facts."\n'
        )
        parsed = self.engine._parse_ask_video_response(llm_output, "What is event-driven architecture?", meta)
        assert "Event-driven architecture provides" in parsed.answer
        assert "KEY EVIDENCE" not in parsed.answer
        assert len(parsed.evidence) == 1
        assert parsed.evidence[0].timestamp_seconds == 600
        assert parsed.evidence[0].speaker_name == "Lead Architect"
        assert parsed.evidence[0].quote == "Events are immutable logs of facts."
