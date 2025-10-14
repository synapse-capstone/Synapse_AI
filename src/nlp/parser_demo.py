def check_nlp_env():
    ok = True
    issues = []
    try:
        import spacy
        nlp = spacy.load("ko_core_news_sm")
        _ = nlp("아이스 아메리카노 하나 주세요.")
    except Exception as e:
        ok = False
        issues.append(f"spaCy or model issue: {e}")
    return ok, issues

if __name__ == "__main__":
    ok, issues = check_nlp_env()
    print("[NLP] ✅ Ready" if ok else "[NLP] ❌ Not ready")
    if issues: print(" -", "\n - ".join(issues))
