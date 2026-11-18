from ml.preprocess import clean_text

def test_removes_html_tags():
    assert clean_text("Great film.<br /><br />Must see!") == "Great film. Must see!"

def test_collapses_spaces():
    assert clean_text("a   lot   cpaces") == "a lot cpaces"

def test_strips_edges():
    assert clean_text("  ok  ") == "ok"