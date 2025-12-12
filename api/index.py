from app import app

def handler(request):
    return app(request.environ, lambda status, headers: None)

