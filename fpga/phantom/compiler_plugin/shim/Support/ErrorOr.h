// Shim: TargetTraits.h names ErrorOr/ErrorOrSuccess only as return types of
// non-virtual members, which the plugin never calls, so declarations suffice.
#pragma once
namespace M {
template <typename T> class ErrorOr;
class ErrorOrSuccess;
} // namespace M
namespace M::KGEN { using M::ErrorOr; using M::ErrorOrSuccess; }
