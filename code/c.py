import world
from utils import sample_ext
print(f"Using C++ sampling: {sample_ext}")
if sample_ext:
    print("✅ Fast C++ negative sampling is active!")
else:
    print("❌ Falling back to slow Python sampling")