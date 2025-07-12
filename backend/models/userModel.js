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
		hasOnboarded: {
			type: Boolean,
			default: false,
		},
		favoriteMovies: {
			type: [String],
			default: [],
		},
		watchlist: {
			type: [String],
			default: [],
		},
		watchedMovies: {
            type: [{
                movieId: {
                    type: String,
                    required: true
                },
                rating: {
                    type: Number,
                    min: 0,
                    max: 5,
                    default: 0 
                },
                watchedAt: {
                    type: Date,
                    default: Date.now
                }
            }],
            default: []
        },
		modelTrainingStatus: {
			lastTrainedAt: {
			type: Date
			},
			ratedMoviesCount: {
			type: Number,
			default: 0
			},
			embeddingsUpdated: {
				type: Boolean,
				default: false
			},
		},
	},
	{ timestamps: true }
);

export const User = mongoose.model("User", userSchema);