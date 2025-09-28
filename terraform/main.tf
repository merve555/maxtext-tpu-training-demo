terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Data source to get project information
data "google_project" "current" {}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs 
resource "google_project_service" "required_apis" {
  for_each = toset([
    "container.googleapis.com",
    "tpu.googleapis.com", 
    "compute.googleapis.com",
    "storage.googleapis.com",
    "artifactregistry.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "file.googleapis.com"
  ])
  
  service = each.value
  disable_on_destroy = false
}

# GKE Standard Cluster with TPU support
resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.region
  
  # Use GKE Standard (not Autopilot) to enable custom TPU node pools
  enable_autopilot = false
  initial_node_count = 1
  
  # Disable deletion protection for demo environment
  deletion_protection = false

  # Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Enable required addons for TPU workloads
  addons_config {
    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
    gcs_fuse_csi_driver_config {
      enabled = true
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
    http_load_balancing {
      disabled = false
    }
  }

  # Network configuration for GKE Standard
  network    = "default"
  subnetwork = "default"

  # Enable IP aliases with automatic range allocation
  ip_allocation_policy {}

  depends_on = [google_project_service.required_apis]
}

# Note: Default node pool is created automatically with the cluster

# DWS Flex TPU node pool for training workloads
# Commented out since manually created in the cluster
/*
resource "google_container_node_pool" "tpu_dws_pool" {
  name     = "tpu-dws-flex-pool"
  cluster  = google_container_cluster.primary.name
  location = var.region

  # Start with 1 nodes
  initial_node_count = 0
  autoscaling {
    min_node_count = 0
    max_node_count = 3 # A reasonable upper limit for your pool
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  # Restrict to us-east5-b zone only where TPU v6e is available
  node_locations = ["us-east5-b"]

  # DWS Flex is activated by the taint, not queued_provisioning

  node_config {
    reservation_affinity {
      consume_reservation_type = "NO_RESERVATION"
    }
    
    # CORRECT: This machine_type is the only field needed to specify the TPU configuration.
    machine_type = "ct6e-standard-8t"

    # CORRECTED: Use a broader scope to allow writing to GCS, logging, etc.
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    service_account = google_service_account.gke_sa.email

    # CORRECTED: This is the official taint for DWS Flex nodes. Your Job YAML
    # must have a toleration for this exact key-value pair to be scheduled here.
    taint {
      key    = "cloud.google.com/gke-dws-flex"
      value  = "true"
      effect = "NO_SCHEDULE"
    }
  }
}
*/

# Service Account for GKE nodes
resource "google_service_account" "gke_sa" {
  account_id   = "merves-gke-sa"
  display_name = "Merves GKE Service Account"
}

# IAM bindings for the service account
resource "google_project_iam_member" "gke_sa_bindings" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
    "roles/tpu.admin",
    "roles/artifactregistry.reader",
    "roles/storage.admin"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_sa.email}"
}

# Artifact Registry for Docker images
resource "google_artifact_registry_repository" "tpu_demo" {
  location      = var.region
  repository_id = "merves-tpu-demo"
  description   = "TPU Training Demo Docker Images"
  format        = "DOCKER"
  
  depends_on = [google_project_service.required_apis]
}

# GCS Bucket for model artifacts
resource "google_storage_bucket" "model_artifacts" {
  name     = "merves-${var.project_id}-${var.cluster_name}-artifacts"
  location = var.region

  uniform_bucket_level_access = true
  
  # Disable deletion protection for demo environment
  force_destroy = true
  
  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
}

# GCS Bucket for datasets
resource "google_storage_bucket" "datasets" {
  name     = "merves-${var.project_id}-${var.cluster_name}-datasets"
  location = var.region

  uniform_bucket_level_access = true
  
  # Disable deletion protection for demo environment
  force_destroy = true
}

resource "google_storage_bucket_iam_member" "gsa_artifacts_access" {
  bucket = google_storage_bucket.model_artifacts.name
  role   = "roles/storage.objectAdmin"
  member = google_service_account.gke_sa.member # This targets the GSA
}

resource "google_storage_bucket_iam_member" "gsa_datasets_access" {
  bucket = google_storage_bucket.datasets.name
  role   = "roles/storage.objectAdmin"
  member = google_service_account.gke_sa.member # This targets the GSA
}

# This binding is CRITICAL for Workload Identity to function.
# It allows your Kubernetes SA to act as your Google SA.
resource "google_project_iam_member" "workload_identity_binding" {
  project = var.project_id
  role    = "roles/iam.workloadIdentityUser"
  
  # This links the GSA to the KSA named "maxtext-sa" in the "default" namespace
  member  = "serviceAccount:${var.project_id}.svc.id.goog[default/maxtext-sa]"
}