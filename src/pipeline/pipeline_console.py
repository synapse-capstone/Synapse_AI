from src.pipeline.pipeline_mock import run_once

print("🗣️  음성인식 시뮬레이터 (텍스트로 대화하세요)")
print("종료하려면 'exit' 또는 '끝' 입력\n")

while True:
    user = input("👤 사용자: ").strip()
    if user.lower() in ["exit", "끝", "종료"]:
        print("🛑 종료합니다.")
        break

    # Whisper 대신 직접 입력된 문장을 NLP로 전달
    response = run_once(user)

    # Google TTS 대신 텍스트로 출력
    print("🤖 키오스크:", response, "\n")
