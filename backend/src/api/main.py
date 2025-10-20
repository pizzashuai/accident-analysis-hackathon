from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from src.api.routes import (
    llm_analysis_route,
    login_route,
    private_route,
    projects_route,
    reports_route,
    users_route,
    video_annotation_route,
)
from src.common.config import settings

api_router = APIRouter()
api_router.include_router(login_route.router)
api_router.include_router(users_route.router)
api_router.include_router(projects_route.router)
api_router.include_router(video_annotation_route.router)
api_router.include_router(llm_analysis_route.router)
api_router.include_router(reports_route.router)

if settings.ENVIRONMENT == "local":
    api_router.include_router(private_route.router)


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
)

# Set all CORS enabled origins
if settings.all_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.all_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)
