output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.primary.name
}

output "cluster_location" {
  description = "GKE cluster location"
  value       = google_container_cluster.primary.location
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "model_artifacts_bucket" {
  description = "GCS bucket for model artifacts"
  value       = google_storage_bucket.model_artifacts.name
}

output "datasets_bucket" {
  description = "GCS bucket for datasets"
  value       = google_storage_bucket.datasets.name
}

output "service_account_email" {
  description = "Service account email for GKE nodes"
  value       = google_service_account.gke_sa.email
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository name"
  value       = google_artifact_registry_repository.tpu_demo.name
}
