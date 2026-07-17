from app.auth.models import AuthenticatedUser, UserRole
from app.security.authorization import AuthorizationService
from app.security.patient_identity import PatientIdentityService


TEST_SECRET = "authorization-test-secret-value-over-32-characters"


def create_service() -> AuthorizationService:
    identity_service = PatientIdentityService(
        TEST_SECRET
    )

    return AuthorizationService(
        identity_service=identity_service
    )


def create_user(
    username: str,
    role: UserRole,
) -> AuthenticatedUser:
    return AuthenticatedUser(
        user_key=f"user-key-{username}",
        username=username,
        display_name="Test User",
        role=role,
    )


def test_patient_can_access_own_record():
    service = create_service()

    user = create_user(
        "P1001",
        UserRole.PATIENT,
    )

    patient = service.get_default_patient(user)

    assert patient is not None

    decision = service.authorize_patient_access(
        user,
        patient.patient_key,
    )

    assert decision.allowed is True
    assert decision.display_alias == "My record"


def test_patient_cannot_access_other_patient():
    identity_service = PatientIdentityService(
        TEST_SECRET
    )

    service = AuthorizationService(
        identity_service=identity_service
    )

    user = create_user(
        "P1001",
        UserRole.PATIENT,
    )

    other_patient_key = (
        identity_service.create_patient_key(
            "P1002"
        )
    )

    decision = service.authorize_patient_access(
        user,
        other_patient_key,
    )

    assert decision.allowed is False
    assert decision.patient_key is None


def test_caregiver_can_access_assigned_patient():
    identity_service = PatientIdentityService(
        TEST_SECRET
    )

    service = AuthorizationService(
        identity_service=identity_service
    )

    user = create_user(
        "C2001",
        UserRole.CAREGIVER,
    )

    patient_key = identity_service.create_patient_key(
        "P1002"
    )

    decision = service.authorize_patient_access(
        user,
        patient_key,
    )

    assert decision.allowed is True


def test_caregiver_cannot_access_unassigned_patient():
    identity_service = PatientIdentityService(
        TEST_SECRET
    )

    service = AuthorizationService(
        identity_service=identity_service
    )

    user = create_user(
        "C2001",
        UserRole.CAREGIVER,
    )

    unassigned_patient_key = (
        identity_service.create_patient_key(
            "P1003"
        )
    )

    decision = service.authorize_patient_access(
        user,
        unassigned_patient_key,
    )

    assert decision.allowed is False


def test_clinician_can_access_assigned_patient():
    identity_service = PatientIdentityService(
        TEST_SECRET
    )

    service = AuthorizationService(
        identity_service=identity_service
    )

    user = create_user(
        "D3001",
        UserRole.CLINICIAN,
    )

    patient_key = identity_service.create_patient_key(
        "P1003"
    )

    decision = service.authorize_patient_access(
        user,
        patient_key,
    )

    assert decision.allowed is True


def test_clinician_cannot_access_unassigned_patient():
    identity_service = PatientIdentityService(
        TEST_SECRET
    )

    service = AuthorizationService(
        identity_service=identity_service
    )

    user = create_user(
        "D3001",
        UserRole.CLINICIAN,
    )

    unassigned_patient_key = (
        identity_service.create_patient_key(
            "P9999"
        )
    )

    decision = service.authorize_patient_access(
        user,
        unassigned_patient_key,
    )

    assert decision.allowed is False


def test_empty_patient_key_is_denied():
    service = create_service()

    user = create_user(
        "P1001",
        UserRole.PATIENT,
    )

    decision = service.authorize_patient_access(
        user,
        "",
    )

    assert decision.allowed is False
    assert decision.reason == "No patient record was selected."


def test_unknown_user_has_no_patient_access():
    service = create_service()

    user = create_user(
        "UNKNOWN",
        UserRole.CAREGIVER,
    )

    assert service.list_authorized_patients(user) == []
    assert service.user_has_patient_access(user) is False


def test_raw_patient_ids_are_not_returned():
    service = create_service()

    user = create_user(
        "C2001",
        UserRole.CAREGIVER,
    )

    patients = service.list_authorized_patients(user)
    serialized = repr(patients)

    assert "P1001" not in serialized
    assert "P1002" not in serialized
    assert "C2001" not in serialized