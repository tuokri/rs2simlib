//=============================================================================
// TypeXXBullet
//=============================================================================
// Bullet for the Type56 (Chinese AK47) Assault Rifle
//=============================================================================
// Test Data for rs2simlib
//=============================================================================
class TypeXXBullet extends ROBullet;

defaultproperties
{
	// BallisticCoefficient=0.225	// 123gr M43 (G1)
	BallisticCoefficient=0.138		// 123gr M43 (G7)

	// Damage=123
	Damage=489
	MyDamageType=class'XXDmgType_TypeXXBullet'
	Speed=36750 // 735 M/S
	MaxSpeed=36750 // 735 M/S

	// RS2. Energy transfer function
	// RPD, Type 56, AKM, SKS
	VelocityDamageFalloffCurve=(Points=((InVal=302760000,OutVal=0.46),(InVal=1560250000,OutVal=0.12)))
	// VelocityDamageFalloffCurve=(Points=((InVal=302760000,OutVal=0.55),(InVal=1560250000,OutVal=0.15)))
	// VelocityDamageFalloffCurve=(Points=((InVal=0.44,OutVal=0.55),(InVal=1.0,OutVal=0.13)))

	DragFunction=RODF_G7
}
