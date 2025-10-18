# FastAPI Project - Backend

## Requirements

- [Docker](https://www.docker.com/).
- [uv](https://docs.astral.sh/uv/) for Python package and environment management.

## Docker Deployment

### Unified Dependency Management for Docker

Both the API and Worker services are built as **separate Docker images** but share the same unified dependency set from the root `pyproject.toml`. Each Dockerfile:

1. Installs the `uv` package manager
2. Copies the root `pyproject.toml` and `uv.lock` files
3. Installs only the required dependencies for that service using optional dependency groups:
   - API image: `uv pip install --system -e ".[api]"`
   - Worker image: `uv pip install --system -e ".[worker]"`
4. Copies the shared `libs/` directory and service-specific code
5. Sets up the appropriate `PYTHONPATH` for imports

This approach ensures:

- **No dependency duplication**: Single source of truth for all package versions
- **Efficient builds**: Docker layer caching for dependencies
- **Smaller images**: Each service only includes its required dependencies
- **Consistency**: Same versions used in development and production

### Docker Compose

Start the local development environment with Docker Compose following the guide in [../development.md](../development.md).

## General Workflow

By default, the dependencies are managed with [uv](https://docs.astral.sh/uv/), go there and install it.

### Unified Dependency Management

This project uses a **unified dependency structure** at the root level. All dependencies are defined in the root `pyproject.toml` with optional dependency groups for each service:

- **Base dependencies**: Shared by all services (sqlmodel, psycopg, pydantic-settings, etc.)
- **API dependencies**: Required only by the API service (`[api]` group)
- **Worker dependencies**: Required only by the worker service (`[worker]` group)
- **Dev dependencies**: Development tools (pytest, mypy, ruff, etc.)

From `./backend/` you can install all the dependencies with:

```console
# Install base + dev dependencies
$ uv sync

# Or install with specific service dependencies
$ uv sync --extra api        # For API development
$ uv sync --extra worker     # For worker development
$ uv sync --all-extras       # Install everything (recommended)
```

Then you can activate the virtual environment with:

```console
$ source .venv/bin/activate
```

Make sure your editor is using the correct Python virtual environment, with the interpreter at `backend/.venv/bin/python`.

### Running Services Locally

You have several options to run the services:

**Option 1: Run both services together (recommended for full-stack development)**

```console
$ ./scripts/dev.sh up
```

**Option 2: Run individual services**

```console
# Run only the API service
$ ./scripts/run-api.sh
# or
$ ./scripts/dev.sh api

# Run only the worker service
$ ./scripts/run-worker.sh
# or
$ ./scripts/dev.sh worker
```

**Option 3: Stop all services**

```console
$ ./scripts/dev.sh down
```

Modify or add SQLModel models for data and SQL tables in `./backend/libs/backend_db/models.py`, API endpoints in `./backend/services/api/routes/`, CRUD (Create, Read, Update, Delete) utils in `./backend/libs/backend_db/crud.py`.

## VS Code

There are already configurations in place to run the backend through the VS Code debugger, so that you can use breakpoints, pause and explore variables, etc.

The setup is also already configured so you can run the tests through the VS Code Python tests tab.

## Docker Compose Override

During development, you can change Docker Compose settings that will only affect the local development environment in the file `docker-compose.override.yml`.

The changes to that file only affect the local development environment, not the production environment. So, you can add "temporary" changes that help the development workflow.

For example, the directory with the backend code is synchronized in the Docker container, copying the code you change live to the directory inside the container. That allows you to test your changes right away, without having to build the Docker image again. It should only be done during development, for production, you should build the Docker image with a recent version of the backend code. But during development, it allows you to iterate very fast.

There is also a command override that runs `fastapi run --reload` instead of the default `fastapi run`. It starts a single server process (instead of multiple, as would be for production) and reloads the process whenever the code changes. Have in mind that if you have a syntax error and save the Python file, it will break and exit, and the container will stop. After that, you can restart the container by fixing the error and running again:

```console
$ docker compose watch
```

There is also a commented out `command` override, you can uncomment it and comment the default one. It makes the backend container run a process that does "nothing", but keeps the container alive. That allows you to get inside your running container and execute commands inside, for example a Python interpreter to test installed dependencies, or start the development server that reloads when it detects changes.

To get inside the container with a `bash` session you can start the stack with:

```console
$ docker compose watch
```

and then in another terminal, `exec` inside the running container:

```console
$ docker compose exec backend bash
```

You should see an output like:

```console
root@7f2607af31c3:/app#
```

that means that you are in a `bash` session inside your container, as a `root` user, under the `/app` directory. The FastAPI code lives under `/app/api` and the database layer under `/app/database/backend_database`.

There you can use the `fastapi run --reload` command to run the debug live reloading server.

```console
$ fastapi run --reload api/main.py
```

...it will look like:

```console
root@7f2607af31c3:/app# fastapi run --reload api/main.py
```

and then hit enter. That runs the live reloading server that auto reloads when it detects code changes.

Nevertheless, if it doesn't detect a change but a syntax error, it will just stop with an error. But as the container is still alive and you are in a Bash session, you can quickly restart it after fixing the error, running the same command ("up arrow" and "Enter").

...this previous detail is what makes it useful to have the container alive doing nothing and then, in a Bash session, make it run the live reload server.

## Backend tests

To test the backend run:

```console
$ bash ./scripts/test.sh
```

The tests run with Pytest, modify and add tests to `./backend/tests/`.

If you use GitHub Actions the tests will run automatically.

### Test running stack

If your stack is already up and you just want to run the tests, you can use:

```bash
docker compose exec backend bash scripts/tests-start.sh
```

That `/app/scripts/tests-start.sh` script just calls `pytest` after making sure that the rest of the stack is running. If you need to pass extra arguments to `pytest`, you can pass them to that command and they will be forwarded.

For example, to stop on first error:

```bash
docker compose exec backend bash scripts/tests-start.sh -x
```

### Test Coverage

When the tests are run, a file `htmlcov/index.html` is generated, you can open it in your browser to see the coverage of the tests.

## Migrations

As during local development your app directory is mounted as a volume inside the container, you can also run the migrations with `alembic` commands inside the container and the migration code will be in your app directory (instead of being only inside the container). So you can add it to your git repository.

Make sure you create a "revision" of your models and that you "upgrade" your database with that revision every time you change them. As this is what will update the tables in your database. Otherwise, your application will have errors.

- Start an interactive session in the backend container:

```console
$ docker compose exec backend bash
```

- Alembic is already configured to import your SQLModel models from `./backend/database/backend_database/models.py`.

- After changing a model (for example, adding a column), inside the container, create a revision, e.g.:

```console
$ alembic revision --autogenerate -m "Add column last_name to User model"
```

- Commit to the git repository the files generated in the alembic directory.

- After creating the revision, run the migration in the database (this is what will actually change the database):

```console
$ alembic upgrade head
```

If you don't want to use migrations at all, uncomment the lines in the file at `./backend/database/backend_database/db.py` that end in:

```python
SQLModel.metadata.create_all(engine)
```

and comment the line in the file `scripts/prestart.sh` that contains:

```console
$ alembic upgrade head
```

If you don't want to start with the default models and want to remove them / modify them, from the beginning, without having any previous revision, you can remove the revision files (`.py` Python files) under `./backend/database/alembic/versions/`. And then create a first migration as described above.

## Email Templates

The email templates are in `./backend/api/email-templates/`. Here, there are two directories: `build` and `src`. The `src` directory contains the source files that are used to build the final email templates. The `build` directory contains the final email templates that are used by the application.

Before continuing, ensure you have the [MJML extension](https://marketplace.visualstudio.com/items?itemName=attilabuti.vscode-mjml) installed in your VS Code.

Once you have the MJML extension installed, you can create a new email template in the `src` directory. After creating the new email template and with the `.mjml` file open in your editor, open the command palette with `Ctrl+Shift+P` and search for `MJML: Export to HTML`. This will convert the `.mjml` file to a `.html` file and now you can save it in the build directory.
