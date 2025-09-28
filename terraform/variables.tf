variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
}

variable "region" {
  description = "Google Cloud region"
  type        = string
  default     = "us-east5"
}

variable "zone" {
  description = "Google Cloud zone (must support TPU v6e)"
  type        = string
  default     = "us-east5-b"
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "merves-tpu-training-demo"
}