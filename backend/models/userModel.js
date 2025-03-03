import mongoose from "mongoose";




const userSchema = new mongoose.Schema(
	{
		email: {
			type: String,
			required: true,
			unique: true,
		},
		password: {
			type: String,
			required: true,
		},
		name: {
			type: String,
			required: true,
		},
		isVerified: {
			type: Boolean,
			default: false,
		},
		hasOnboarded: {
			type: Boolean,
			default: false,
		},
		favoriteGenres: {
			type: [String],
			default: [],
		},
		favoriteMovies: {
			type: [String],
			default: [],
		},

		streamingServices: {
		type: [String],
		default: [],
		},
		movieType: {
			type: [String],
			default: [],
		},


		
		
		resetPasswordToken: String,
		resetPasswordExpiresAt: Date,
		verificationToken: String,
		verificationTokenExpiresAt: Date,
	},
	{ timestamps: true }
);

export const User = mongoose.model("User", userSchema);