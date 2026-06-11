from __future__ import annotations

import numpy as np

from embed_tree import (
    FunctionLabeler,
    HuggingFaceTextEmbedder,
    LabelCandidate,
    LabelRequest,
    LLMConfig,
    LLMLabeler,
    PCAConfig,
    PCAProjector,
    embed_texts,
)


class FakeSentenceTransformer:
    def encode(self, texts, convert_to_numpy=True, **kwargs):
        assert convert_to_numpy is True
        return np.asarray([[len(t), len(t) + 1] for t in texts], dtype=np.float32)


def test_huggingface_text_embedder_uses_injected_model():
    embedder = HuggingFaceTextEmbedder(model_obj=FakeSentenceTransformer(), device="cpu")

    assert embedder.device == "cpu"
    assert np.allclose(embedder.embed("abc"), [3, 4])
    assert np.allclose(embedder.embed_batch(["a", "abcd"]), [[1, 2], [4, 5]])
    assert np.allclose(embed_texts(embedder, ["xy"]), [[2, 3]])


def test_pca_projector_fit_transform_round_trip():
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ]
    )
    projector = PCAProjector(PCAConfig(dims=2))

    projector.fit(vectors)
    reduced = projector(vectors)
    restored = PCAProjector.from_dict(projector.to_dict())

    assert reduced.shape == (4, 2)
    assert restored.is_fitted
    assert np.allclose(restored.transform(vectors), reduced)


def test_function_labeler_streams_user_function():
    req = LabelRequest(
        candidates=[
            LabelCandidate(id="a", text="auth token refresh"),
            LabelCandidate(id="b", text="auth session login"),
        ]
    )
    labeler = FunctionLabeler(lambda request: (chunk for chunk in ["auth", " session"]))

    assert list(labeler.stream(req)) == ["auth", " session"]
    assert labeler.label(req) == "auth session"


def test_llm_labeler_wraps_existing_tagger_client():
    class Message:
        content = "Auth sessions"

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    class Completions:
        def create(self, **kwargs):
            return Response()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    req = LabelRequest(candidates=[LabelCandidate(id="a", text="refresh login token")])
    labeler = LLMLabeler(LLMConfig(provider="openai"), client=Client())

    assert list(labeler.stream(req)) == ["Auth sessions"]
    assert labeler.label(req) == "Auth sessions"
