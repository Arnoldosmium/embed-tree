"""Tagger unit tests: KeywordTagger, LLMTagger (stubbed), make_tagger."""

from embed_tree import KeywordTagger, LLMConfig, LLMTagger, make_tagger


def test_keyword_tagger_picks_distinctive_term():
    label = KeywordTagger(top_k=2)(["python code", "python script", "python module"])
    assert "python" in label


def test_keyword_tagger_empty():
    assert KeywordTagger()([]) == ""
    assert KeywordTagger()(["", "   "]) == ""


# --- OpenAI LLM tagger via injected stub client (no network) ---------------

class _Msg:
    def __init__(self, c):
        self.content = c


class _Choice:
    def __init__(self, c):
        self.message = _Msg(c)


class _Resp:
    def __init__(self, c):
        self.choices = [_Choice(c)]


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


def test_llm_tagger_openai_cleans_label():
    client = StubClient('"Animal Topics."\nextra line')
    tagger = LLMTagger(LLMConfig(provider="openai", max_label_words=6), client=client)
    label = tagger(["the cat", "a dog"])
    assert label == "Animal Topics"  # quotes/period/extra-lines stripped
    assert client.chat.completions.last_kwargs["model"] == "gpt-4o-mini"


def test_llm_tagger_respects_max_words():
    client = StubClient("one two three four five six seven eight")
    tagger = LLMTagger(LLMConfig(provider="openai", max_label_words=3), client=client)
    assert tagger(["x"]) == "one two three"


def test_llm_tagger_local_pipeline():
    def stub_pipe(prompt, **kwargs):
        return [{"generated_text": "Food items\nignored"}]

    tagger = LLMTagger(LLMConfig(provider="local"), pipeline=stub_pipe)
    assert tagger(["pizza", "sushi"]) == "Food items"


def test_make_tagger_dispatch():
    assert isinstance(make_tagger(LLMConfig(provider="none")), KeywordTagger)
    t = make_tagger(LLMConfig(provider="openai"), client=StubClient("x"))
    assert isinstance(t, LLMTagger)
