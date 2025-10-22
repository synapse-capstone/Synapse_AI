from src.dialogue.manager import DialogueCtx, next_turn

print("🎛 음성 키오스크 상태머신 시뮬레이터 (텍스트 입력)")
ctx = DialogueCtx()
resp = next_turn(ctx, "")   # BOOT -> GREETING
print("🤖:", resp)

while True:
    user = input("👤 ").strip()
    if user in ["exit","끝","종료"]: break
    resp = next_turn(ctx, user)
    print("🤖:", resp)
