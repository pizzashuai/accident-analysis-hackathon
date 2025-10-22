#!/usr/bin/env bash
set -euo pipefail

# Deploy the background worker to AWS App Runner in us-east-2 (override via AWS_REGION).
# Relies on AWS CLI v2, Docker, and Python for env parsing.

command -v aws >/dev/null 2>&1 || {
  echo "aws CLI not found. Install AWS CLI v2 before running this script." >&2
  exit 1
}

command -v docker >/dev/null 2>&1 || {
  echo "Docker not found. Install and start Docker before running this script." >&2
  exit 1
}

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "python3 (or python) not found. Install Python to process environment files." >&2
    exit 1
  fi
fi

usage() {
  local exit_code="${1:-1}"
  echo "Usage: $0 [-r REGION|--region REGION] [-f DOCKERFILE|--dockerfile DOCKERFILE] [-c CONTEXT|--context CONTEXT]" >&2
  echo "          [-p PLATFORM|--platform PLATFORM] [-e FILE|--env-file FILE] [--no-env-file]" >&2
  exit "${exit_code}"
}

AWS_REGION="${AWS_REGION:-us-east-2}"
SERVICE_NAME="${SERVICE_NAME:-accident-worker}"
ECR_REPO="${ECR_REPO:-accident-worker}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"
HEALTHCHECK_PATH="${HEALTHCHECK_PATH:-/health}"
HEALTHCHECK_PROTOCOL="${HEALTHCHECK_PROTOCOL:-HTTP}"
CPU_SIZE="${CPU_SIZE:-1 vCPU}"
MEMORY_SIZE="${MEMORY_SIZE:-2 GB}"
ROLE_NAME="${ROLE_NAME:-AppRunnerECRAccessRole}"
POLICY_NAME="${POLICY_NAME:-AppRunnerECRAccessPolicy}"
BUILD_CONTEXT="${BUILD_CONTEXT:-.}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-src/worker/Dockerfile}"
DOCKER_TARGET="${DOCKER_TARGET:-}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
ENV_FILE="${ENV_FILE:-.env.prod}"
INCLUDE_ENV_FILE="${INCLUDE_ENV_FILE:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--region)
      [[ $# -ge 2 ]] || usage
      AWS_REGION="$2"
      shift 2
      ;;
    -f|--dockerfile)
      [[ $# -ge 2 ]] || usage
      DOCKERFILE_PATH="$2"
      shift 2
      ;;
    -c|--context)
      [[ $# -ge 2 ]] || usage
      BUILD_CONTEXT="$2"
      shift 2
      ;;
    -e|--env-file)
      [[ $# -ge 2 ]] || usage
      ENV_FILE="$2"
      INCLUDE_ENV_FILE=1
      shift 2
      ;;
    --no-env-file)
      INCLUDE_ENV_FILE=0
      shift
      ;;
    -p|--platform)
      [[ $# -ge 2 ]] || usage
      DOCKER_PLATFORM="$2"
      shift 2
      ;;
    -h|--help)
      usage 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      ;;
  esac
done

AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
REPO_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"
ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${ROLE_NAME}"
runtime_env_json=""
runtime_env_property=""
healthcheck_path_entry=""

HC_PROTOCOL_UPPER="$(printf '%s' "${HEALTHCHECK_PROTOCOL}" | tr '[:lower:]' '[:upper:]')"
if [[ "${HC_PROTOCOL_UPPER}" != "HTTP" && "${HC_PROTOCOL_UPPER}" != "TCP" ]]; then
  echo "Unsupported health check protocol: ${HEALTHCHECK_PROTOCOL}. Use HTTP or TCP." >&2
  exit 1
fi
HEALTHCHECK_PROTOCOL="${HC_PROTOCOL_UPPER}"
if [[ "${HEALTHCHECK_PROTOCOL}" == "HTTP" ]]; then
  healthcheck_path_entry=$(printf ',\n    "Path": "%s"' "${HEALTHCHECK_PATH}")
fi

if [[ "${INCLUDE_ENV_FILE}" == "1" ]]; then
  if [[ -f "${ENV_FILE}" ]]; then
    echo "Loading runtime environment variables from ${ENV_FILE}"
    runtime_env_json="$("${PYTHON_BIN}" - "${ENV_FILE}" <<'PY'
import json
import os
import sys

path = sys.argv[1]
data = {}
with open(path, encoding="utf-8") as fh:
    for raw_line in fh:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        data[key] = value
json.dump(data, sys.stdout, separators=(",", ":"))
PY
)"
    runtime_env_json="${runtime_env_json//$'\n'/}"
    if [[ "${runtime_env_json}" == "{}" ]]; then
      runtime_env_json=""
    fi
  else
    echo "Environment file ${ENV_FILE} not found; skipping runtime environment variables." >&2
  fi
fi

if [[ -n "${runtime_env_json}" ]]; then
  runtime_env_property=$(printf ',\n        "RuntimeEnvironmentVariables": %s' "${runtime_env_json}")
fi

echo "Using AWS account: ${AWS_ACCOUNT_ID}"
echo "AWS region: ${AWS_REGION}"
echo "App Runner service name: ${SERVICE_NAME}"
echo "ECR repository: ${ECR_REPO}"
echo "Image tag: ${IMAGE_TAG}"
echo "Container port: ${CONTAINER_PORT}"
echo "Health check protocol: ${HEALTHCHECK_PROTOCOL}"
if [[ "${HEALTHCHECK_PROTOCOL}" == "HTTP" ]]; then
  echo "Health check path: ${HEALTHCHECK_PATH}"
else
  echo "Health check path not used for protocol ${HEALTHCHECK_PROTOCOL}"
fi
echo "Docker build platform: ${DOCKER_PLATFORM}"
if [[ "${INCLUDE_ENV_FILE}" == "1" ]]; then
  if [[ -n "${runtime_env_json}" ]]; then
    echo "Runtime environment variables loaded from: ${ENV_FILE}"
  else
    echo "Env file ${ENV_FILE} provided no variables (or is empty after filtering)."
  fi
else
  echo "Runtime environment variables disabled via --no-env-file."
fi

echo "Ensuring ECR repository ${ECR_REPO} exists..."
if ! aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "${ECR_REPO}" --region "${AWS_REGION}"
fi

echo "Authenticating Docker to ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

if [[ "${SKIP_DOCKER_BUILD:-0}" != "1" ]]; then
  if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
    echo "Dockerfile not found at ${DOCKERFILE_PATH}. Set DOCKERFILE_PATH env or pass --dockerfile." >&2
    exit 1
  fi
  if [[ ! -d "${BUILD_CONTEXT}" ]]; then
    echo "Build context directory not found at ${BUILD_CONTEXT}. Set BUILD_CONTEXT env or pass --context." >&2
    exit 1
  fi
  echo "Building Docker image..."
  docker build \
    --platform "${DOCKER_PLATFORM}" \
    --file "${DOCKERFILE_PATH}" \
    ${DOCKER_TARGET:+--target "${DOCKER_TARGET}"} \
    --tag "${SERVICE_NAME}" \
    "${BUILD_CONTEXT}"
else
  echo "Skipping docker build because SKIP_DOCKER_BUILD=1"
fi

if [[ "${SKIP_DOCKER_TAG_PUSH:-0}" != "1" ]]; then
  echo "Tagging image as ${REPO_URI}..."
  docker tag "${SERVICE_NAME}:latest" "${REPO_URI}"

  echo "Pushing image to ECR..."
  docker push "${REPO_URI}"
else
  echo "Skipping docker tag/push because SKIP_DOCKER_TAG_PUSH=1"
fi

trust_file="$(mktemp)"
policy_file="$(mktemp)"
create_file="$(mktemp)"
update_file="$(mktemp)"
trap 'rm -f "${trust_file}" "${policy_file}" "${create_file}" "${update_file}"' EXIT

cat >"${trust_file}" <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "build.apprunner.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

cat >"${policy_file}" <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    }
  ]
}
EOF

echo "Ensuring IAM role ${ROLE_NAME} exists..."
if ! aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document "file://${trust_file}"
fi

aws iam update-assume-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-document "file://${trust_file}"

aws iam put-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-name "${POLICY_NAME}" \
  --policy-document "file://${policy_file}"

SERVICE_ARN="$(aws apprunner list-services \
  --region "${AWS_REGION}" \
  --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceArn" \
  --output text 2>/dev/null || true)"

cat >"${create_file}" <<EOF
{
  "ServiceName": "${SERVICE_NAME}",
  "SourceConfiguration": {
    "ImageRepository": {
      "ImageIdentifier": "${REPO_URI}",
      "ImageRepositoryType": "ECR",
      "ImageConfiguration": {
        "Port": "${CONTAINER_PORT}"${runtime_env_property}
      }
    },
    "AuthenticationConfiguration": {
      "AccessRoleArn": "${ROLE_ARN}"
    },
    "AutoDeploymentsEnabled": false
  },
  "InstanceConfiguration": {
    "Cpu": "${CPU_SIZE}",
    "Memory": "${MEMORY_SIZE}"
  },
  "HealthCheckConfiguration": {
    "Protocol": "${HEALTHCHECK_PROTOCOL}"${healthcheck_path_entry},
    "Interval": 10,
    "HealthyThreshold": 1,
    "UnhealthyThreshold": 5,
    "Timeout": 5
  },
  "NetworkConfiguration": {
    "IngressConfiguration": {
      "IsPubliclyAccessible": true
    },
    "EgressConfiguration": {
      "EgressType": "DEFAULT"
    }
  }
}
EOF

cat >"${update_file}" <<EOF
{
  "SourceConfiguration": {
    "ImageRepository": {
      "ImageIdentifier": "${REPO_URI}",
      "ImageRepositoryType": "ECR",
      "ImageConfiguration": {
        "Port": "${CONTAINER_PORT}"${runtime_env_property}
      }
    },
    "AuthenticationConfiguration": {
      "AccessRoleArn": "${ROLE_ARN}"
    },
    "AutoDeploymentsEnabled": false
  },
  "InstanceConfiguration": {
    "Cpu": "${CPU_SIZE}",
    "Memory": "${MEMORY_SIZE}"
  },
  "HealthCheckConfiguration": {
    "Protocol": "${HEALTHCHECK_PROTOCOL}"${healthcheck_path_entry},
    "Interval": 10,
    "HealthyThreshold": 1,
    "UnhealthyThreshold": 5,
    "Timeout": 5
  },
  "NetworkConfiguration": {
    "IngressConfiguration": {
      "IsPubliclyAccessible": true
    },
    "EgressConfiguration": {
      "EgressType": "DEFAULT"
    }
  }
}
EOF

if [[ -z "${SERVICE_ARN}" || "${SERVICE_ARN}" == "None" ]]; then
  echo "Creating App Runner service ${SERVICE_NAME}..."
  aws apprunner create-service \
    --region "${AWS_REGION}" \
    --cli-input-json "file://${create_file}"
else
  echo "Updating App Runner service ${SERVICE_NAME}..."
  aws apprunner update-service \
    --region "${AWS_REGION}" \
    --service-arn "${SERVICE_ARN}" \
    --cli-input-json "file://${update_file}"
fi

echo "Deployment request submitted. Check service status with:"
echo "  aws apprunner list-services --region ${AWS_REGION}"
echo "Fetch service URL:"
echo "  aws apprunner describe-service --service-arn <service-arn> --query 'Service.ServiceUrl' --output text --region ${AWS_REGION}"
