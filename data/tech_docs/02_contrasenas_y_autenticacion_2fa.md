# Gestión de Contraseñas y Autenticación de Dos Factores (2FA) — i2T

## Objetivo

Asegurar el uso correcto de credenciales corporativas y del segundo factor de autenticación, protegiendo el acceso a los sistemas de información de i2T y de sus clientes.

## Responsabilidades

- **Soporte i2T:** administra el directorio de usuarios, resetea contraseñas y configura el segundo factor de autenticación.
- **Colaborador:** mantiene sus credenciales en forma confidencial y actualiza su contraseña según la política vigente.

## Política de contraseñas

- La contraseña corporativa debe tener un mínimo de 10 caracteres, combinando mayúsculas, minúsculas, números y al menos un carácter especial.
- Se solicita el cambio de contraseña cada 90 días. El sistema notifica con 7 días de anticipación al vencimiento.
- No se permite reutilizar las últimas 5 contraseñas utilizadas.
- Está prohibido anotar la contraseña en papel, compartirla por chat o enviarla por correo electrónico sin cifrar.

## Cómo cambiar la contraseña del correo corporativo

1. Ingresar al portal de autoservicio de i2T con el usuario corporativo.
2. Seleccionar la opción "Cambiar contraseña".
3. Ingresar la contraseña actual y la nueva contraseña, cumpliendo con la política vigente.
4. Confirmar el cambio; el sistema sincroniza automáticamente el correo, SuiteCRM y JIRA en un plazo de hasta 15 minutos.

Si el colaborador no recuerda su contraseña actual, debe solicitar un reseteo a soporte@i2t.com.ar indicando su nombre completo y usuario corporativo. Por seguridad, el reseteo solo puede solicitarse desde el correo personal registrado en el legajo o telefónicamente con validación de identidad.

## Autenticación de dos factores (2FA)

Todos los sistemas críticos de i2T (correo corporativo, VPN, SuiteCRM y JIRA) requieren autenticación de dos factores mediante una aplicación de autenticación (Google Authenticator o Microsoft Authenticator).

### Problemas frecuentes con el doble factor de autenticación

- **Perdí el celular donde tenía configurada la aplicación de autenticación:** el colaborador debe notificar de inmediato a soporte@i2t.com.ar para desactivar el 2FA anterior y volver a vincular un nuevo dispositivo, presentando validación de identidad.
- **El código de verificación no funciona:** generalmente se debe a un desfase de horario entre el dispositivo móvil y el servidor. Se recomienda sincronizar la hora automática del celular antes de reintentar.
- **Cambié de celular y no migré la aplicación de autenticación:** Soporte puede generar un nuevo código de configuración (QR) para vincular el segundo factor al nuevo dispositivo, previa validación de identidad del colaborador.

## Nota de calidad

Al abandonar el repositorio electrónico de documentos, el presente deja de ser un documento válido y vigente del Sistema de Gestión de Calidad.
