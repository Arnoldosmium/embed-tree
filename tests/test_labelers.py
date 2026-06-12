"""Labeler unit tests."""

from embed_tree import KeywordLabeler, LLMConfig, LLMLabeler, LabelCandidate, LabelRequest


def request(*texts: str, max_words: int = 6) -> LabelRequest:
    return LabelRequest(candidates=[LabelCandidate(id=i, text=text) for i, text in enumerate(texts)], max_words=max_words)


def test_keyword_labeler_picks_distinctive_term():
    label = KeywordLabeler(top_k=2).label(request("python code", "python script", "python module"))
    assert "python" in label


def test_keyword_labeler_empty():
    assert KeywordLabeler().label(request()) == ""
    assert KeywordLabeler().label(request("", "   ")) == ""


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, text):
        self._text = text
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _Resp(self._text)


class _Chat:
    def __init__(self, text):
        self.completions = _Completions(text)


class StubClient:
    def __init__(self, text):
        self.chat = _Chat(text)


def test_llm_labeler_openai_cleans_label():
    client = StubClient('"Animal Topics."\nextra line')
    labeler = LLMLabeler(LLMConfig(provider="openai", max_label_words=6), client=client)
    label = labeler.label(request("the cat", "a dog"))
    assert label == "Animal Topics"
    assert client.chat.completions.last_kwargs["model"] == "gpt-4o-mini"


def test_llm_labeler_respects_max_words():
    client = StubClient("one two three four five six seven eight")
    labeler = LLMLabeler(LLMConfig(provider="openai", max_label_words=3), client=client)
    assert labeler.label(request("x", max_words=3)) == "one two three"


def test_llm_labeler_local_pipeline():
    def stub_pipe(prompt, **kwargs):
        return [{"generated_text": "Food items\nignored"}]

    labeler = LLMLabeler(LLMConfig(provider="local"), pipeline=stub_pipe)
    assert labeler.label(request("pizza", "sushi")) == "Food items"
