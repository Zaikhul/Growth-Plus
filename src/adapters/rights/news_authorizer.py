"""Map reviewed news operations to the existing rights port; unknown grants deny."""

from src.domain.rights import DataOperation
from src.ports.rights_authorizer import RightsAuthorizer


class AuthorizedNewsRights:
    def __init__(
        self,
        authorizer: RightsAuthorizer,
        dataset_id: str,
        rights_version: str,
        collection_operation: DataOperation,
        retention_operation: DataOperation,
    ) -> None:
        if not dataset_id or not rights_version:
            raise ValueError("Reviewed dataset and rights version are required")
        self.authorizer, self.dataset_id, self.rights_version = (
            authorizer,
            dataset_id,
            rights_version,
        )
        self.collection_operation, self.retention_operation = (
            collection_operation,
            retention_operation,
        )

    def _check(self, operation: DataOperation) -> str:
        result = self.authorizer.authorize("gdelt", self.dataset_id, operation)
        if not result.allowed:
            raise PermissionError("GDELT operation denied by registered rights policy")
        return self.rights_version

    def assert_collection(self) -> str:
        return self._check(self.collection_operation)

    def assert_retention(self) -> str:
        return self._check(self.retention_operation)

    def assert_feature_use(self) -> str:
        return self._check(DataOperation.COMPUTE_FEATURES)
