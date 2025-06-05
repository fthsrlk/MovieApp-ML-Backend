"""WSGI giriş noktası: Waitress veya herhangi bir WSGI sunucusu bu dosyayı kullanarak
 uygulamayı üretim modunda başlatabilir.

Kullanım (örnek, Windows için):
```
pip install waitress
set USE_PRODUCTION_SERVER=1
waitress-serve --port=8000 --call 'wsgi:create_app'
```
"""

import os
from typing import Any

from waitress import serve  # type: ignore

# Flask uygulamasını döndüren factory fonksiyonu

def create_app() -> Any:
    # Üretim moduna geçtiğimizi belirtmek için env değişkeni ayarla
    os.environ.setdefault("USE_PRODUCTION_SERVER", "1")

    # app.py içindeki global `app` tanımını içe aktar
    from app import app  # noqa: WPS433 (run-time import intentional)

    return app

if __name__ == "__main__":
    # Lokal olarak "python wsgi.py" ile de sunucuyu başlatabilmek için
    serve(create_app(), host="0.0.0.0", port=8000) 